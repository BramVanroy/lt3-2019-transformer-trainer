from collections.abc import Iterable, Iterator, MutableMapping
from copy import deepcopy
from itertools import product
import json
import logging
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class Config(Iterator):
    instance = None
    OPTIMIZABLE_PARAMETERS = {'model': ['dropout', 'pre_classifier'],
                              'optimizer': ['lr', 'weight_decay', 'eps']}

    def __init__(self, config_file, local_rank, distributed):
        if Config.instance is not None:
            raise LookupError('Only one instance of Config can exist at a time.')
        current_path = Path(__file__).resolve().parent
        # Overwrite defaults with custom config
        with current_path.joinpath('defaults.json').open(encoding='utf-8') as defaults_fh, \
                open(config_file, encoding='utf-8') as config_fh:
            config = self._recursive_merge(json.load(defaults_fh), json.load(config_fh))

        # Sanity check: is num_labels same as n target_names
        if config['training']['num_labels'] > 1 and config['training']['target_names']:
            if len(config['training']['target_names']) != config['training']['num_labels']:
                raise ValueError("The given number of 'target_names' is not identical to 'num_labels'.")

        # not part of self.config! Don't want this in output config.json
        self.process = None
        self._set_process_variables(local_rank, distributed)
        self.device = self.process['device']

        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)


        # Only use fp16 if we're actually running on the GPU
        config['training']['fp16'] = config['training']['fp16'] if self.device.type == 'cuda' else False
        config['training']['metric_name'] = 'MSE' if config['training']['task'] == 'regression' else 'CrossEntropy'
        config['training']['metric_name2nd'] = 'pearson' if config['training']['task'] == 'regression' else 'f1'
        config['training']['f1_average'] = self._set_f1_average(config)

        # Save the original config (listified)
        self._listified_config = self._listify_config(config)
        self._params = {k: {param: self._listified_config[k][param]
                            for param in params if param in self._listified_config[k]}
                        for k, params in self.OPTIMIZABLE_PARAMETERS.items()}

        # Keep track of the changing config, that changes for each param combination
        # Temporarily set config so it can be used to init Predictor
        self.model = config['model']
        self.optimizer = None
        self.training = config['training']
        self.config = config

        self._gen = self._generate_config()
        self.iteration = 0
        self._n_combos = len(list(self._generate_combinations(self._params)))
        Config.instance = self

    @classmethod
    def get(cls):
        return cls.instance

    def _generate_combinations(self, d):
        """ Generates dict combinations of all non-dict values at any depth
            Borrowed from https://stackoverflow.com/a/50606871/1150683 """
        keys, values = d.keys(), d.values()
        values_choices = (self._generate_combinations(v) if isinstance(v, dict) else v for v in values)

        for comb in product(*values_choices):
            yield dict(zip(keys, comb))

    def _generate_config(self):
        """ Builds all possible combinations from the parameters that are allowed to be combined.
            These are defined in OPTIMIZABLE_PARAMETERS. """
        for combo in self._generate_combinations(self._params):
            config = self._recursive_merge(deepcopy(self._listified_config), combo)
            self.config = config
            self.model = self.config['model']
            self.optimizer = self.config['optimizer']
            self.scheduler = self.config['scheduler']
            self.training = self.config['training']
            self.iteration += 1
            yield config

    def _listify_config(self, config):
        """ Turn parameters into list when we expect them to be. """
        config_copy = deepcopy(config)
        for k, pars in self.OPTIMIZABLE_PARAMETERS.items():
            for par in pars:
                if par not in config_copy[k]:
                    continue
                obj = config_copy[k][par]
                if isinstance(obj, str) or not isinstance(obj, Iterable):
                    config_copy[k][par] = [obj]

        return config_copy

    def _recursive_merge(self, d1, d2):
        """ Updates d1 with values from d2 without overwriting.
            Borrowed from https://stackoverflow.com/a/24088493/1150683 """
        for k, v in d1.items():
            if k in d2 and all(isinstance(e, MutableMapping) for e in (v, d2[k])):
                d2[k] = self._recursive_merge(v, d2[k])
        d3 = d1.copy()
        d3.update(d2)
        return d3

    @staticmethod
    def _set_f1_average(config):
        num_labels = config['training']['num_labels']
        f1_average = config['training']['f1_average']
        if num_labels == 1:
            f1_average = None
        else:
            if f1_average == 'auto':
                f1_average = 'binary' if num_labels == 2 else 'micro'

        return f1_average

    def _set_process_variables(self, local_rank, distributed):
        """ Set current device to use. If gpu_id is false-y, the CPU will be used. """

        world_size = 1
        if local_rank == -1 or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device(f"cuda:{local_rank}")
            if distributed:
                dist.init_process_group(backend='nccl', init_method='env://')
                world_size = dist.get_world_size()

        self.process = {
            'device': device,
            'distributed': distributed,
            'is_first': not distributed or local_rank in [0, -1],
            'local_rank': local_rank,
            'world_size': world_size
        }

    def __len__(self):
        return self._n_combos

    def __iter__(self):
        return self._gen

    def __next__(self):
        try:
            return next(self._gen)
        except StopIteration:
            return None
