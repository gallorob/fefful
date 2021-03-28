# evocraft-life-project

This repository contains the code for the [Minecraft Open-Endedness Challenge](https://evocraft.life/).

The algorithm presented is *FEFFuL: Few Examples Fitness Function Learner*, a custom way aimed at reducing user fatigue to a minimum in IEC OEE systems.

## Installation
We suggest using [Conda](https://anaconda.org/) to create a virtual environment.
- Install Conda on your machine
- Download this repository: `git clone https://github.com/gallorob/evocraft-life-project.git`
- Modify `evo_env.yml`, setting the `Prefix` variable to your local conda installation path
  - You can also change the environment name if it clashes with a preexisting one
- From within the repository folder, create the Conda environment: `conda env create -f evo_env.yml`
- You can then activate the environment using `conda activate evo`
- Download the [EvoCraft API](https://github.com/real-itu/Evocraft-py) repository (you shouldn't need to install any additional dependency) and follow their installation instructions.
  - Optional: test if everything works by running their `example.py`
- Download the [PyTorch Neat](https://github.com/gallorob/PyTorch-NEAT) repository (you shouldn't need to install any additional dependency) and follow their installation instructions.
  - Optional: test if everything works by running their examples in the `examples` folder