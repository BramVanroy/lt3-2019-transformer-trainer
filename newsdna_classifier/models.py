from collections import defaultdict
from copy import deepcopy
from functools import partial
from importlib import import_module
import math

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, GPT2Tokenizer

from Config import Config

MODELS = {
    'albert': 'albert-base-v1',
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'gpt2': 'gpt2',
    'openai': 'openai-gpt',
    'roberta': 'roberta-base',
    'transfoxl': 'transfo-xl-wt103',
    'xlm': 'xlm-mlm-enfr-1024',
    'xlnet': 'xlnet-base-cased'
}

try:
    # CTRL has been added in transformers 2.1.0
    MODELS['ctrl'] = 'ctrl'
    # T5 will be added in 2.3.0 (?)
    MODELS['t5'] = 't5-small'
except NameError:
    pass


# Define custom activation functions
def gelu_new(x):
    """ Different gelu implementation. See https://arxiv.org/abs/1606.08415 """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    """ See https://arxiv.org/abs/1710.05941 """
    return x * torch.sigmoid(x)


try:
    # Released with PyTorch 1.2
    from torch.nn.functional import gelu
except AttributeError:
    def gelu(x):
        """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
            For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerClassifier(nn.Module):
    """ Main transformer class that can initialize any kind of transformer in `MODELS`. """

    def __init__(self):
        super(TransformerClassifier, self).__init__()
        self.config = Config.get()
        self.mopts = self.config.model
        self.topts = self.config.training

        self.distributed = self.config.process['distributed']
        self.local_rank = self.config.process['local_rank']

        if self.mopts['name'] == 'transfoxl':
            raise NotImplementedError("The 'transfoxl' model is currently not implemented.")

        # Only download the required files once, in the first process
        # See https://github.com/huggingface/transformers/issues/1521#issuecomment-542105087
        if self.distributed and self.local_rank != 0:
            torch.distributed.barrier()

        self.weights = MODELS[self.mopts['name']] if self.mopts['weights'] == 'default' else self.mopts['weights']
        model_config = AutoConfig.from_pretrained(self.weights, output_hidden_states=True)
        self.base_model = AutoModel.from_pretrained(self.weights, config=model_config)
        if self.distributed and self.local_rank == 0:
            torch.distributed.barrier()

        self.n_concat_layers = len(self.mopts['layers'])

        # Freeze parts of pretrained model
        # config['freeze'] can be "all" to freeze all layers,
        # or any number of prefixes, e.g. ['embeddings', 'encoder']
        if self.mopts['freeze']:
            for name, param in self.base_model.named_parameters():
                if self.mopts['freeze'] == 'all' \
                        or 'all' in self.mopts['freeze'] \
                        or name.startswith(tuple(self.mopts['freeze'])):
                    param.requires_grad = False

        dim = self.base_model.config.hidden_size
        pooled_dim = dim * self.n_concat_layers

        if pooled_dim == 0:
            raise ValueError("Because of your config, the output dimension is (unacceptably) zero.")

        if self.mopts['pre_classifier'] == 'auto':
            self.mopts['pre_classifier'] = pooled_dim

        if self.mopts['pre_classifier']:
            self.pre_classifier = nn.Linear(pooled_dim, self.mopts['pre_classifier'])
        else:
            self.pre_classifier = None

        if self.mopts['activation'] and 'name' in self.mopts['activation'] and self.mopts['activation']['name']:
            self.activation = self._get_activation()
        else:
            self.activation = None

        if self.mopts['dropout']:
            self.dropout = nn.Dropout(self.mopts['dropout'])
        else:
            self.dropout = None

        self.classifier = nn.Linear(self.mopts['pre_classifier'] if self.mopts['pre_classifier'] else pooled_dim,
                                    self.config.training['num_labels'])

        self.tokenizer_wrapper = TransformerTokenizer(self)
        self.cls_token_id = self.tokenizer_wrapper.tokenizer.cls_token_id

        if self.cls_token_id is None:
            raise ValueError("The tokenizer must have cls_token_id")

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None):
        if self.mopts['name'] in {'albert', 'bert', 'xlm', 'xlnet'}:
            out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        if self.mopts['name'] in {'ctrl', 'distilbert', 'openai', 't5', 'xlm', 'xlnet'}:
            hidden_states = out[1]
        else:
            # for 'albert', 'bert', 'gpt2', 'roberta', 'transfoxl'
            hidden_states = out[2]

        if self.n_concat_layers == 1:
            output = hidden_states[self.mopts['layers'][0]]
        else:
            output = torch.cat([hidden_states[i] for i in self.mopts['layers']], dim=-1)

        output = output[torch.where(input_ids == self.cls_token_id)]

        if self.pre_classifier:
            output = self.pre_classifier(output)

        if self.activation:
            output = self.activation(output)

        if self.dropout:
            output = self.dropout(output)

        clf = self.classifier(output)

        return clf

    def _get_activation(self):
        cfg_copy = deepcopy(self.mopts['activation'])
        act_name = cfg_copy.pop('name')
        if act_name == 'gelu':
            return gelu
        elif act_name == 'gelu_new':
            return gelu_new
        elif act_name == 'swish':
            return swish
        else:
            func = getattr(import_module('torch.nn.functional'), act_name)
            return partial(func, **cfg_copy)


class TransformerTokenizer:
    """ Main tokenizer class that can initialize any kind of transformer in `MODELS`. """

    def __init__(self, model):
        self.model = model
        self.config = Config.get()
        self.mopts = self.config.model
        self.topts = self.config.training

        self.distributed = self.config.process['distributed']
        self.local_rank = self.config.process['local_rank']

        # Only download the required files once, in the first process
        if self.distributed and self.local_rank != 0:
            torch.distributed.barrier()

        self.tokenizer = AutoTokenizer.from_pretrained(model.weights)
        if self.distributed and self.local_rank == 0:
            torch.distributed.barrier()

        if self.mopts['name'] in {'ctrl', 'gpt2', 'openai'}:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]', 'pad_token': '[PAD]'})
            self.model.base_model.resize_token_embeddings(len(self.tokenizer))

        if self.mopts['name'] in {'albert', 'bert', 'distilbert', 'xlm', 'xlnet'}:
            self.n_separating_tokens = 1
        elif self.mopts['name'] in {'roberta'}:
            self.n_separating_tokens = 2

        self.needs_prefix_space = isinstance(self.tokenizer, GPT2Tokenizer)

    def encode_batch_plus(self,
                          batch,
                          batch_pair=None,
                          pad_to_batch_length=False,
                          return_tensors=None,
                          return_token_type_ids=True,
                          return_attention_mask=True,
                          return_special_tokens_mask=False,
                          **kwargs):
        if pad_to_batch_length and 'pad_to_max_length' in kwargs and kwargs['pad_to_max_length']:
            raise ValueError("'pad_to_batch_length' and 'pad_to_max_length' cannot be used simultaneously.")

        # ensure that encoding happens the same as when we get the mask_position
        if self.needs_prefix_space:
            kwargs['add_prefix_space'] = True

        def merge_dicts(list_of_ds):
            d = defaultdict(list)
            for _d in list_of_ds:
                for _k, _v in _d.items():
                    d[_k].append(_v)

            return dict(d)

        # gather all encoded inputs in a list of dicts
        encoded = []
        batch_pair = [None] * len(batch) if batch_pair is None else batch_pair
        for first_sent, second_sent in zip(batch, batch_pair):
            # return_tensors=None: don't convert to tensors yet. Do that manually as the last step
            encoded.append(self.tokenizer.encode_plus(first_sent,
                                                      second_sent,
                                                      return_tensors=None,
                                                      return_token_type_ids=return_token_type_ids,
                                                      return_attention_mask=return_attention_mask,
                                                      return_special_tokens_mask=return_special_tokens_mask,
                                                      max_length=self.mopts['max_seq_len'],
                                                      **kwargs))

        # convert list of dicts in a single merged dict
        encoded = merge_dicts(encoded)

        if pad_to_batch_length:
            max_batch_len = max([len(l) for l in encoded['input_ids']])

            if self.tokenizer.padding_side == 'right':
                if return_attention_mask:
                    encoded['attention_mask'] = [mask + [0] * (max_batch_len - len(mask))
                                                 for mask in encoded['attention_mask']]
                if return_token_type_ids:
                    encoded["token_type_ids"] = [ttis + [self.tokenizer.pad_token_type_id] * (max_batch_len - len(ttis))
                                                 for ttis in encoded['token_type_ids']]
                if return_special_tokens_mask:
                    encoded['special_tokens_mask'] = [stm + [1] * (max_batch_len - len(stm))
                                                      for stm in encoded['special_tokens_mask']]
                encoded['input_ids'] = [ii + [self.tokenizer.pad_token_id] * (max_batch_len - len(ii))
                                        for ii in encoded['input_ids']]
            elif self.tokenizer.padding_side == 'left':
                if return_attention_mask:
                    encoded['attention_mask'] = [[0] * (max_batch_len - len(mask)) + mask
                                                 for mask in encoded['attention_mask']]
                if return_token_type_ids:
                    encoded['token_type_ids'] = [[self.tokenizer.pad_token_type_id] * (max_batch_len - len(ttis))
                                                 for ttis in encoded['token_type_ids']]
                if return_special_tokens_mask:
                    encoded['special_tokens_mask'] = [[1] * (max_batch_len - len(stm)) + stm
                                                      for stm in encoded['special_tokens_mask']]

                encoded['input_ids'] = [[self.tokenizer.pad_token_id] * (max_batch_len - len(ii)) + ii
                                        for ii in encoded['input_ids']]

        if return_tensors is not None:
            if return_tensors == 'pt':
                encoded['input_ids'] = torch.tensor(encoded['input_ids'])
                if 'attention_mask' in encoded:
                    encoded['attention_mask'] = torch.tensor(encoded['attention_mask'])
                if 'token_type_ids' in encoded:
                    encoded['token_type_ids'] = torch.tensor(encoded['token_type_ids'])
                if 'special_tokens_mask' in encoded:
                    encoded['special_tokens_mask'] = torch.tensor(encoded['special_tokens_mask'])
            else:
                raise ValueError(f"Cannot return tensors with value '{return_tensors}'")

        return encoded
