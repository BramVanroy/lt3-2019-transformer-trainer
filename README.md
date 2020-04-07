Transformers classification
===========================

**Code will not be updated. This is legacy code that works perfectly fine, but that lacks good documentation. We are working on a great interface to create your own Trainers over at [HuggingFace](https://github.com/huggingface/transformers). This repository was made public to showcase how to allow for a configurable user experience, using automatic mixed precision, using distributed learning, using early stopping, and so on.**

Remainder of the documentation is intended for in-house usage at LT3. 

Requirements
------------
- At least Python 3.6
- PyTorch, preferably with CUDA support. Have a look at th PyTorch [Getting Started](https://pytorch.org/get-started/locally/)
page on how to install torch for our environment. (You don't need to install torchvision.) Weoh has CUDA10.1 installed.
If you run into issues with the torch installation, you can come see me or ask Michaël. This is likely due to the default
CUDA version not being CUDA10.1.
- Apart from PyTorch you also need: 
  - pandas
  - sklearn
  - scipy
  - matplotlib
  - transformers
 
**Recommended**: simply do `pipenv install` in the root of this directory. It should install all the dependencies above.
`torch` needs to be installed separately, though. Easiest is finding the correct command on the PyTorch website for `pip`.
If the command given is simply `pip install torch`, you can just do `pipenv install torch`. If it is something like the 
following, **don't just copy-paste and execute** because it will install torch globally which you probably don't want. 

```bash
pip install torch===1.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**Instead**, you need to ensure to install it in the virtual environment that you created above with `pipenv`.

Option 1:

```bash
pipenv run python -m pip install torch===1.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Option 2:

```bash
pipenv shell
python -m pip install torch===1.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

How to use
----------
- The only thing you should ever change, is the configuration file `config.json`. Particularly the `num_labels`,
`files` and `output_dir`. Don't change any other files.
- Your input files should contain three named columns: `id` (a given sentence ID), `text` (the actual text), `label`
(the correct label index for the given sentence). By default, the script expects a tab-separated file, but you can 
change the separator in the config file by adding `"sep": "yourseperator"` to the `training` section.
- **Do not train on your laptop.** This will take billions times longer than running it on the server.
- Some parameters accept a list of arguments. This will do a naive hyperparameter search, i.e. all possible combinations
will be tried out. For example, you can put `"dropout": [0, 0.2]`, which implies that the script will be ran two times
with different dropout values. You can mix-and-match, too! E.g. in addition to the dropout list, you can set 
`"pre_classifier": [null, 2048, 1536]`. All possible combinations will be tried out, so with this combination of dropout
and pre_classifier, that will lead to six different runs. All results will be saved the the given `output_dir`. At the 
end of all runs, the script will tell you which combination was best, both in terms of minimal loss or maximal secondary
score (F1 or Pearson).


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
