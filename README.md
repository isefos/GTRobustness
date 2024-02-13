# TODO: update README
### Python environment setup with Conda

```bash
conda create -n gtr python=3.11
conda activate gtr

# install the version of pytorch that is compatible with your system
# e.g.:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# or for cpu only:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch

conda install -c pyg pyg pytorch-scatter

conda install -c conda-forge yacs tensorboardx lightning torchmetrics performer-pytorch ogb wandb

pip install seml

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
# conda install openbabel fsspec rdkit -c conda-forge

conda clean --all
```



### W&B logging
To use W&B logging, set `wandb.use True` and have a `gtransformers` entity set-up in your W&B account (or change it to whatever else you like by setting `wandb.entity`).



## Unit tests

To run all unit tests, execute from the project root directory:

```bash
python -m unittest -v
```

Or specify a particular test module, e.g.:

```bash
python -m unittest -v unittests.test_eigvecs
```
