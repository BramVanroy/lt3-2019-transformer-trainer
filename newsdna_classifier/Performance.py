from collections import defaultdict
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, classification_report
import torch
import torch.distributed as dist

from Config import Config

logger = logging.getLogger(__name__)


class Performance:
    def __init__(self):
        self.config = Config.get()
        self.popts = self.config.process
        self.topts = self.config.training

        self.device = self.popts['device']
        self.distributed = self.popts['distributed']

        self.metric_name2nd = self.topts['metric_name2nd']
        self.num_labels = self.topts['num_labels']
        self.multi_label = bool(self.topts['multi_label'])
        self.task = self.topts['task']
        self.patience = self.topts['patience']

        self.min_valid_loss = np.inf
        self.last_saved_epoch = 0

        self._start_time = 0
        self.training_time = 0

        self._tensors = None
        self.gathered = {}
        self._averages = defaultdict(list)
        self._init_tensors()

    def clear(self):
        self._init_tensors()

    def end_train_loop_time(self):
        self.training_time += time.time() - self._start_time

    def evaluate(self, epoch):
        done_training = False
        save_model = False

        avg_train_loss = torch.mean(self.gathered['train']['losses']).to(self.device)
        avg_valid_loss = torch.mean(self.gathered['valid']['losses']).to(self.device)

        self._averages['train'].append(avg_train_loss)
        self._averages['valid'].append(avg_valid_loss)

        train_2nd_metric = self._evaluate_2nd_metric('train')
        valid_2nd_metric = self._evaluate_2nd_metric('valid')

        # Log epoch statistics
        logger.info(f"Epoch {epoch:,}"
                    f"\nTraining:   loss {avg_train_loss:.6f} - {self.metric_name2nd}: {train_2nd_metric}"
                    f"\nValidation: loss {avg_valid_loss:.6f} - {self.metric_name2nd}: {valid_2nd_metric}")

        if avg_valid_loss <= self.min_valid_loss:
            logger.info(f"!! Validation loss decreased ({self.min_valid_loss:.6f} --> {avg_valid_loss:.6f}).")
            self.last_saved_epoch = epoch
            self.min_valid_loss = avg_valid_loss
            save_model = True
        else:
            logger.info(f"!! Valid loss not improved. (Min. = {self.min_valid_loss:.6f};"
                        f" last save at ep. {self.last_saved_epoch})")
            if avg_train_loss <= avg_valid_loss:
                logger.warning(f"!! Training loss is lte validation loss. Might be overfitting!")

        # Early-stopping
        if self.patience and (epoch - self.last_saved_epoch) == self.patience:
            logger.info(f"Stopping early at epoch {epoch} (patience={self.patience})...")
            done_training = True
        elif epoch == self.config.training['epochs']:
            done_training = True

        fig = None
        if done_training:
            fig = self._plot_training()

        return avg_valid_loss, valid_2nd_metric, fig, save_model, done_training

    def _evaluate_2nd_metric(self, partition):
        """ Evaluates results of the 2nd metric. Pearson for regression, f1 for classification. """
        # In the rare case where all results are identical, metric calculation might fail
        # return None in that case
        with np.errstate(all='raise'):
            try:
                if self.task == 'regression':
                    # Returns a tuple: (r, p-value)
                    if self.num_labels == 1:
                        metric_res = pearsonr(self.gathered[partition]['labels'],
                                              self.gathered[partition]['preds'])[0]
                    else:
                        pearsonrs = []
                        for i in range(self.num_labels):
                            labels_i = self.gathered[partition]['labels'][:, i].tolist()
                            preds_i = self.gathered[partition]['preds'][:, i].tolist()
                            r_i = pearsonr(labels_i, preds_i)
                            pearsonrs.append(r_i[0])
                        # Average pearson-r of all labels
                        metric_res = np.mean(pearsonrs)
                else:
                    # Returns a float
                    metric_res = f1_score(self.gathered[partition]['labels'],
                                          self.gathered[partition]['preds'],
                                          average=self.topts['f1_average'])
            except FloatingPointError:
                metric_res = None

        return metric_res

    def evaluate_test(self):
        avg_test_loss = torch.mean(self.gathered['test']['losses']).item()
        test_2nd_metric = self._evaluate_2nd_metric('test')

        report = None
        if self.task == 'classification':
            report = classification_report(self.gathered['test']['labels'],
                                           self.gathered['test']['preds'],
                                           target_names=self.topts['target_names'])
            logger.info(f"Classification report:\n{report}")

        # If regression, return the first item of the tuple (r, p-value)
        # If classification, return number (f1)
        return avg_test_loss, test_2nd_metric if self.task == 'regression' else test_2nd_metric, report

    def _init_tensors(self):
        tensor_constructor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
        self._tensors = {part: defaultdict(tensor_constructor) for part in ('test', 'train', 'valid')}

        if self.task == 'classification':
            # for classification, we need a LongTensor rather than float
            for part in ('test', 'train', 'valid'):
                if not self.multi_label:
                    self._tensors[part]['labels'] = torch.cuda.LongTensor([]) if self.device.type == 'cuda' else torch.LongTensor([])
                    self._tensors[part]['preds'] = torch.cuda.LongTensor([]) if self.device.type == 'cuda' else torch.LongTensor([])
                    if part == 'test':
                        self._tensors[part]['probs'] = torch.cuda.FloatTensor([]) if self.device.type == 'cuda' else torch.FloatTensor([])
                else:
                    self._tensors[part]['labels'] = torch.cuda.FloatTensor([]) if self.device.type == 'cuda' else torch.FloatTensor([])
                    self._tensors[part]['preds'] = torch.cuda.FloatTensor([]) if self.device.type == 'cuda' else torch.FloatTensor([])

        for part in ('test', 'train', 'valid'):
            self._tensors[part]['sentence_ids'] = torch.cuda.LongTensor([]) if self.device.type == 'cuda' else torch.LongTensor([])

    def gather_cat(self, *partitions):
        if self.distributed:
            for partition in partitions:
                self.gathered[partition] = {
                    'losses': self._gather_cat('losses', partition).cpu(),
                    'labels': self._gather_cat('labels', partition).cpu(),
                    'sentence_ids': self._gather_cat('sentence_ids', partition).cpu(),
                    'preds': self._gather_cat('preds', partition).cpu()
                }
        else:
            self.gathered = {part: {attr: tensor.cpu()
                                    for attr, tensor in d.items()}
                             for part, d in self._tensors.items()}

    def _gather_cat(self, attr, partition):
        x = self._tensors[partition][attr]
        gather = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gather, x)
        return torch.cat(gather)

    def _plot_training(self):
        """ Plot loss into plt graph.
            :returns the figure object of the graph """
        train_losses = self._averages['train']
        valid_losses = self._averages['valid']
        fig = plt.figure(dpi=300)
        plt.plot(train_losses, label='Training loss')
        plt.plot(valid_losses, label='Validation loss')
        plt.xlabel('epochs')
        plt.legend(frameon=False)
        plt.title(f"Loss progress ({self.topts['metric_name']})")
        # Set ticks to integers for the epochs rather than floats
        plt.xticks(ticks=range(len(train_losses)), labels=range(1, len(train_losses)+1))
        plt.show()

        return fig

    def start_train_loop_time(self):
        self._start_time = time.time()

    def update(self, attr, partition, tensor):
        # with batch_size 1, the prediction tensor might be zero-dimensions
        if not tensor.size():
            tensor = tensor.unsqueeze(0)
        if not self.multi_label:
            self._tensors[partition][attr] = torch.cat((self._tensors[partition][attr], tensor.detach()))
        else:
            if attr == 'sentence_ids':
                self._tensors[partition][attr] = torch.cat((self._tensors[partition][attr], tensor.detach()))
            else:
                self._tensors[partition][attr] = torch.cat((self._tensors[partition][attr], tensor.float().detach()))
