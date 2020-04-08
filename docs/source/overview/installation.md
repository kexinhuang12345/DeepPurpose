# Installation

Users can clone and build the repository locally or install Selene through conda/pip. 

Please use Selene with Python 3.6+.

**Install [PyTorch](https://pytorch.org/get-started/locally/).** If you have an NVIDIA GPU, install a version of PyTorch that supports it--Selene will run much faster with a discrete GPU.

## Installing with Anaconda

To install with conda (recommended for Linux users), run the following command in your terminal:
```
conda install -c bioconda selene-sdk
```

### Installing selene with pip:

```sh
pip install selene-sdk
```

Note that we do not recommend pip-installing older versions of Selene (below 0.4.0), as these releases were less stable. 

We currently only have a source distribution available for pip-installation. We are looking into releasing wheels in the future. 

## Installing from source

Selene can also be installed from source.
First, download the latest commits from the source repository:
```
git clone https://github.com/FunctionLab/selene.git
```

The `setup.py` script requires NumPy, Cython, and setuptools. Please make sure you have these already installed.

If you plan on working in the `selene` repository directly, we recommend [setting up a conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using `selene-cpu.yml` or `selene-gpu.yml` (if CUDA is enabled on your machine) and activating it.

Selene contains some Cython files. You can build these by running
```sh
python setup.py build_ext --inplace
```

If you would like to locally install Selene, you can run
```sh
python setup.py install
```

## Additional dependency for running the CLI 

Please install `docopt` before running the command-line script `selene_cli.py` provided in the repository.

