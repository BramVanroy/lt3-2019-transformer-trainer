Base LT3 Transformer classifier 2019-2020
=========================================

**!!Code will not be updated!!**

This is legacy code that works perfectly fine, but that lacks good documentation and 
that lacks a consistent approach to inference and testing as well as model save/loading.
I recommend using the easily adapatable Trainers over at [HuggingFace](https://github.com/huggingface/transformers).
This repository was made public to showcase how to allow for a configurable user experience, using automatic mixed
precision, using distributed training, using early stopping, and so on.

Remainder of the documentation is intended for in-house usage at LT3. This code has been used for many different research
topics within LT3.

Using it in your own work?
--------------------------
**Great!** Glad to hear that this repository has been of use to you. If you publish code based on this repository,
*always* include the provided LICENSE file in your copy/modification of the code (and optionally link to this
repository from your own). Also, if you produce and publish research results by use of (a modification of) this
repository, please place a reference (link) to this repository in your publication.

Requirements
------------
- At least Python 3.6
- PyTorch, preferably with CUDA support. Have a look at th PyTorch [Getting Started](https://pytorch.org/get-started/locally/)
page on how to install torch for our environment. (You don't need to install torchvision or torchaudio.) If you run
  into issues with the torch installation, you can come see me or ask Michaël. 
- Apart from PyTorch you also need: 
  - pandas
  - sklearn
  - scipy
  - matplotlib
  - transformers


How to use
----------
- The only thing you should ever change, is the configuration file `config.json`. Particularly the `task`, `num_labels`,
`multi_label`, `files` and `output_dir`. Don't change any other files unless you know what you are doing. Do not change
  defaults.json. It contains useful default values that are merged with your custom configuration file. If you need a 
  base config file to start from, ask me.
- Your input files should contain three named columns: `id` (a given sentence ID), `text` (the actual text), `label`
(the correct label index for the given sentence). By default, the script expects a tab-separated file, but you can 
change the separator in the config file by adding `"sep": "yourseperator"` to the `training` section.
- **Do not train on your laptop.** This will take billions times longer than running it on the server.
- Some parameters accept a list of arguments. This will do a naive hyperparameter search, i.e. all possible combinations
will be tried out. For example, you can put `"dropout": [0, 0.2]`, which implies that the script will be run two times
with different dropout values. You can mix-and-match, too! E.g. in addition to the dropout list, you can set 
`"pre_classifier": [null, 2048, 1536]`. All possible combinations will be tried out, so with this combination of dropout
and pre_classifier, that will lead to six different runs. All results will be saved the the given `output_dir`. At the 
end of all runs, the script will tell you which combination was best, both in terms of minimal loss or maximal secondary
score (F1 or Pearson).

Multi-label classification
--------------------------

Multi-label classification (and regression) was first added by [Luna De Bruyne](https://github.com/LunaDeBruyne). 
 You can use it by setting the option `multi_label` to `true` in your config file
 under the training options. In such a scenario and when using categorical classification, it is expected that you
 have encoded your categories 0 and 1. Both for regression and classification, you must set `num_labels` to the number
 of labels *per item* that you are predicting, so if you have three categories that are encoded as 0/1, e.g. `1,0,1` 
 in your data file, then `num_labels` should be 3 (not 2).

In terms of data, the columns should still be `id`, `text`, and `label` as above but for multi-label classification, 
 the labels should be separated by commas in the `label` column.

Running the script
------------------
Again: **do not run this on your laptop**.

Because fine-tuning a language model takes a while, you may want to run this in `screen` mode, meaning that you can 
leave the process running on the server without your laptop's terminal being on as well. To start a new, named screen:

```bash
screen -S myscreenname
```

You can leave this screen ('detach') at any time by pressing `CTRL+a+d`. You can resume ('attach') the screen's session
by typing 

```bash
screen -rad myscreenname
```

You can completely terminate a screen by typing `exit`. (But remember  that this is the same command as for exiting a
virtual environment. So if you are inside a screen and inside a virtual environment in that screen, you have to `exit`
twice.)

After installation of the virtual environment, you can run the script on multiple GPUs on Weoh. This is great, because
PyTorch scales really well - meaning that training will be a lot faster.

**Be aware of other users on the server!** You don't want to hog the server while others are working on it, too. You can 
check if a GPU is used by typing `nvidia-smi`. The bottom table shows active processes.

If all GPUs are free, you can run the following code (preferably in your newly created screen session):

```bash
# activate environment in root dir
pipenv shell
# 4 because we have 4 GPUs
python -m torch.distributed.launch --nproc_per_node 4 newsdna_classifier/predict.py <your-config-file>
```

**However**, if one or more GPUs are already being used, you need to tell your terminal that you only wish to "see" a
subset of available devices. This can be done by the `CUDA_VISIBLE_DEVICES` environment variable. As an example, if
you see that the first GPU is taken (GPU ID=0), then we can only use 1, 2 and 3.

```bash
# 1,2,3 because we don't want to use 0
# 3 because we can only use 3 GPUs
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node 3 newsdna_classifier/predict.py <your-config-file>
```

By default, the script will use the CPU. If you want to use the GPU, you explicitly have to pass `--local_rank 0`, or
 similar, e.g.

```bash
# Only make GPU ID #2 visible to the script, which internally then will be rank 0
CUDA_VISIBLE_DEVICES=2 python newsdna_classifier/predict.py <your-config-file> --local_rank 0
```

**Inference** is possible by passing the path to the saved model. In such a case, predictions will be created for the
 test file in your config file. These predictions will be saved to `predictions.csv` by default, but the path can be
 changed by setting `pred_output_path` in the config file under `training` options.

```bash
python newsdna_classifier/predict.py <your-config-file> --infer data/saved/best-model-chkpnt.pth --local_rank 0
```

Simply **testing** the performance of a model according to the `test` file in the config, is also easy. Running a
 commando similar to the one below will test the outputs of a given models compared with the correct labels. The
 command line will print the results to the screen but not save them.

```bash
python newsdna_classifier/predict.py <your-config-file> --test data/saved/best-model-chkpnt.pth --local_rank 0
```
