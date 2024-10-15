# Ontogeny of behavior

This repository contains the code and notebooks for reproducing figures and analyses from "Ontogeny of rodent behavior".

## Installation

Create a new conda environment and install python >= 3.8

```bash
conda create -n ontogeny python=3.10
```

Navigate to the `ontogeny` folder, and install:

```bash
cd /path/to/ontogeny

# run this if you plan on changing the python scripts in the `aging/` folder
pip install -e ".[all]"

# run this if you just plan on using the scripts without modifying them
pip install ".[all]"
```

There are a few other installation options.
1. Install without jax or pytorch: `pip install .`
2. Install with jax: `pip install ".[jax]"`
3. Install with pytorch: `pip install ".[torch]"`

This installs the "aging" python package, along with jax and pytorch dependencies.
Often there are cuda conflicts between jax and pytorch, so it is recommended to install only one of the two.
There are parts of this project that rely on jax and others that rely on pytorch, but those parts of the project are largely isolated.
That means if you only install jax or pytorch, you won't have import errors when running the jax/pytorch-specific parts of the project.

When in doubt, `pip install -e ".[all]"` is the safest option.

## Repository structure

### `aging/` folder

This folder contains the python package which contains the code for the project.
You need to `pip install` it in order to import the code in the notebooks. See [above](#installation).

### `notebooks/` folder

This folder contains exploratory notebooks (in the `exploration/` folder), data processing notebooks (in the `processing/` folder), and figure-generating notebooks (in the `figures/` folder).
The `exploration/` and `figures/` folders contain sub-folders with for Dana's or Win's notebooks.

The notebooks in the `figures/` folder often make one or more figure panels are should be clearly labeled with the figure they are associated with.

### `nextflow/` folder

This folder contains code for running multiple data processing pipelines with nextflow.
Pipelines include:

- depth data extraction, size-normalization, and processing
- MoSeq model application and experiment organization
- training size-normalization models, including hyperparameters scans