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
pip install -e .

# run this if you just plan on using the scripts without modifying them
pip install .
```

When in doubt, `pip install -e .` is the safest option.