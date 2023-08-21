# tf-melt
TF-MELT (TensorFlow Machine Learning Toolbox) is a collection of architectures, processing, and utilities that are transferable over a range of ML applications


## Environment

First, create a new conda environment and activate:

`conda create -n tf-melt`

`conda activate tf-melt`

Next, install pip which will automatically install severl necessary packages:

`conda install pip`

Finally, install the `tfmelt` as a package through pip either through a local install from a git clone

### Local git clone

If you cloned the repo and would like to install from the local git repo, navigate to the head directory where `setup.py` is located and type:

`pip install .`

If you want to update the pip install to make sure dependencies are current:

`pip install --upgrade .`

### Directly from github

To install the `tfmelt` package directly from github simply type:

pip install git+https://github.com/nickwimer/tf-melt.git


## Contributing

pip install black isort flake8