import argparse
import gc
from copy import deepcopy
from importlib import import_module
import json
import logging
from math import inf
import os
from pathlib import Path
import random
from shutil import move

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

try:
    from apex import amp
    try:
        from apex.parallel import convert_syncbn_model
        APEX_PARALLEL_AVAILABLE = True
    except AttributeError:
        APEX_PARALLEL_AVAILABLE = False
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    APEX_PARALLEL_AVAILABLE = False

from Config import Config
from models import TransformerClassifier
from TransformerTrainer import TransformerTrainer


logger = logging.getLogger(__name__)


class TransformerPredictor:
    """ Entry point to start training.
        * Initializes and trains models in `predict()`
        * Tests a model in `test()` """
    def __init__(self):
        self.config = Config.get()

        self.distributed = self.config.process['distributed']
        self.local_rank = self.config.process['local_rank']
        self.world_size = self.config.process['world_size']
        self.device = self.config.process['device']

        self.output_p = Path(self.config.training.pop('output_dir')).resolve()
        self.output_p.mkdir(exist_ok=True, parents=True)

    def clear(self):
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def infer(self, model_f):
        model = TransformerClassifier()
        model, _, map_location = self._optimize_model(model)

        trainer = TransformerTrainer(model=model,
                                     map_location=map_location)

        trainer.infer(checkpoint_f=model_f)

        if self.distributed:
            dist.destroy_process_group()

    def predict(self):
        """ All possible combinations will be tested.
            Will save the best model checkpoint for each configuration as well as its configuration file,
            and a loss graph. If classification, also a classification report. Saved to 'output_dir' in config.
        """
        best_loss = (inf, None)
        best_metric = (0, None)
        # Config changes on each iteration
        while next(self.config):
            logger.info(f"Training parameter combination {self.config.iteration}/{len(self.config)}")
            self._set_seed()

            model = TransformerClassifier()
            optimizer, optim_name = self._get_optim(model)
            scheduler = self._get_scheduler(optimizer)

            model, optimizer, map_location = self._optimize_model(model, optimizer)

            trainer = TransformerTrainer(model=model,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         map_location=map_location)
            # train
            fig, loss, metric_output = trainer.train()
            del model, optimizer, scheduler
            self.clear()

            report = None
            if trainer.datasets['test']:
                loss, metric_output, report = trainer.test('tmp-checkpoint.pth')
                self.clear()

            if self.config.process['is_first']:
                s = self._get_output_prefix(loss, metric_output)
                s = self._check_existence(s)

                model_out = self.output_p.joinpath(s+'model.pth')

                if loss < best_loss[0] or self.config.iteration == 1:
                    best_loss = (loss, model_out)
                    logging.info(f"!! New best loss ({self.config.training['metric_name']}"
                                 f" {best_loss[0]:.4f}): {best_loss[1]}")
                if metric_output > best_metric[0] or self.config.iteration == 1:
                    best_metric = (metric_output, model_out)
                    logging.info(f"!! New highest metric ({self.config.training['metric_name2nd']}"
                                 f" {best_metric[0]:.4f}): {best_metric[1]}")

                # write config file based on actual values
                with self.output_p.joinpath(s+'config.json').open('w', encoding='utf-8') as fhout:
                    json.dump(self.config.config, fhout)

                # move output model to output_dir
                move('tmp-checkpoint.pth', model_out)
                # Save plot
                fig.savefig(self.output_p.joinpath(s+'plot.png'))

                if report:
                    with self.output_p.joinpath(s+'report.txt').open('w', encoding='utf-8') as fhout:
                        fhout.write(report)

            if self.distributed:
                dist.barrier()

        logging.info(f"Done processing all {len(self.config)} parameters combinations.\n")
        logging.info(f"Model with smallest loss ({self.config.training['metric_name']} {best_loss[0]:.4f})"
                     f": {best_loss[1]}")
        logging.info(f"Model with highest metric ({self.config.training['metric_name2nd']} {best_metric[0]:.4f}):"
                     f" {best_metric[1]}")

        if self.distributed:
            dist.destroy_process_group()

    def test(self, model_f):
        model = TransformerClassifier()
        model, _, map_location = self._optimize_model(model)

        trainer = TransformerTrainer(model=model,
                                     map_location=map_location)

        trainer.test(checkpoint_f=model_f)

        if self.distributed:
            dist.destroy_process_group()

    def _check_existence(self, s):
        """ Because only so much information can be summarized in a file name, it is possible that different
            configurations end up with the exact same filename. Therefore, if a file name already exists,
            a new version is saved by adding a number to its filename. """
        counter = 0
        s_temp = s
        while self.output_p.joinpath(s_temp + 'model.pth').exists():
            s_temp = deepcopy(s)
            counter += 1
            s_temp += f"{counter}-"

        return s_temp

    def _get_optim(self, model):
        """ Get the optimizer based on current config. Tries to import name from
            torch.optim.lr_scheduler or transformers.optimization. """
        optim_copy = deepcopy(self.config.optimizer)
        optim_name = optim_copy.pop('name', None)

        try:
            optim_constructor = getattr(import_module('torch.optim'), optim_name)
        except AttributeError:
            try:
                optim_constructor = getattr(import_module('transformers.optimization'), optim_name)
            except AttributeError:
                try:
                    optim_constructor = getattr(import_module('apex.optimizers'), optim_name)
                except (AttributeError, ImportError):
                    raise AttributeError(f"Optimizer '{optim_name}' not found in 'torch', 'transformers', or 'apex'")
        logging.info(f"Using optimizer '{optim_name}'")

        param_optimizer = list(model.named_parameters())
        no_decay = optim_copy.pop('no_decay')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': optim_copy.pop('weight_decay')},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]

        optimizer = optim_constructor(optimizer_grouped_parameters, **optim_copy)

        return optimizer, optim_name

    def _get_output_prefix(self, loss, metric_output):
        """ Create output prefix for the output files based on current config and results """
        mopts = self.config.model
        oopts = self.config.optimizer
        sopts = self.config.scheduler
        topts = self.config.training

        # results
        s = f"loss{loss:.2f}-{topts['metric_name2nd']}{metric_output:.2f}-"
        # optimizer
        s += f"{oopts['name']}-lr{oopts['lr']:.0E}-"
        # model
        s += f"{mopts['name']}-"
        if 'pre_classifier' in mopts and mopts['pre_classifier']:
            s += f"preclfr{mopts['pre_classifier']}-"
        if mopts['activation'] and 'name' in mopts['activation'] and mopts['activation']['name']:
            s += f"{mopts['activation']['name']}-"
        if 'dropout' in mopts and mopts['dropout']:
            s += f"drop{mopts['dropout']:.2f}-"
        # scheduler
        if 'name' in sopts and sopts['name']:
            s += f"{sopts['name']}-"

        return s

    def _get_scheduler(self, optimizer):
        """ Get the scheduler based on current config. Tries to import name from
            torch.optim.lr_scheduler or transformers.optimization. """
        try:
            sched_copy = deepcopy(self.config.scheduler)
            sched_name = sched_copy.pop('name', None)
        except KeyError:
            sched_name = None

        if sched_name:
            try:
                sched_constructor = getattr(import_module('torch.optim.lr_scheduler'), sched_name)
            except AttributeError:
                try:
                    sched_constructor = getattr(import_module('transformers.optimization'), sched_name)
                except AttributeError:
                    raise AttributeError(f"Scheduler '{sched_name}' not found in 'torch' or 'transformers'")
            logging.info(f"Using scheduler '{sched_name}'")
            return sched_constructor(optimizer, **sched_copy)
        else:
            return None

    def _init_amp(self, model, optimizer=None):
        model = model.to(self.device)
        fp16 = self.config.training['fp16']
        if fp16 and optimizer:
            if not AMP_AVAILABLE:
                raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')

            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16)

        return model, optimizer

    def _optimize_model(self, model, optimizer=None):
        model, optimizer = self._init_amp(model, optimizer)

        map_location = None
        if self.distributed:
            if APEX_PARALLEL_AVAILABLE:
                model = convert_syncbn_model(model)

            n = torch.cuda.device_count() // self.world_size
            device_ids = list(range(self.local_rank * n, (self.local_rank + 1) * n))
            rank0_devices = [x - self.local_rank * len(device_ids) for x in device_ids]
            map_location = {f"cuda:{x}": f"cuda:{y}" for x, y in zip(rank0_devices, device_ids)}

            # for some models, DistributedDataParallel might complain about parameters
            # not contributing to loss. find_used_parameters remedies that.
            model = DistributedDataParallel(model,
                                            device_ids=device_ids,
                                            output_device=device_ids[0],
                                            find_unused_parameters=True)

        return model, optimizer, map_location

    def _set_seed(self):
        """ Set all seeds to make results reproducible (deterministic mode).
            When seed is None, disables deterministic mode. """
        seed = self.config.training['seed']

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    cparser = argparse.ArgumentParser(description='Train regression or classifier model with transformers.')
    cparser.add_argument('config_f', help='Path to JSON file with configuration options.')
    cparser.add_argument('--local_rank', default=-1, type=int)
    cparser.add_argument('--test',
                         help='Path to saved model to test. Ensure that the model architecture is the same as the'
                              ' one defined in the provided configuration file.',
                         default=None)
    cparser.add_argument('--infer',
                         help="Path to saved model to test or 'default'. Ensure that the model architecture is the same as the"
                              " one defined in the provided configuration file. The predictions for the file in"
                              " 'test' will be saved.",
                         default=None)

    cargs = cparser.parse_args()

    # torch.distributed.launch adds a world_size environment variable
    distrib = int(os.environ['WORLD_SIZE']) > 1 if 'WORLD_SIZE' in os.environ else False

    # Setup logging for this process
    logging.basicConfig(format='%(asctime)s - [%(levelname)s]: %(message)s',
                        datefmt='%d-%b %H:%M:%S',
                        level=logging.INFO if not distrib or cargs.local_rank in [-1, 0] else logging.WARN)

    config = Config(cargs.config_f, cargs.local_rank, distrib)
    predictor = TransformerPredictor()

    if cargs.test:
        predictor.test(cargs.test)
    elif cargs.infer:
        predictor.infer(cargs.infer)
    else:
        predictor.predict()
