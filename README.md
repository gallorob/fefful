# evocraft-life-project

This repository contains the code for the [Minecraft Open-Endedness Challenge](https://evocraft.life/).

The algorithm presented is *FEFFuL: Few Examples Fitness Function Learner*, a custom way aimed at reducing user fatigue to a minimum in IEC OEE systems.

## Installation
We suggest using [Conda](https://anaconda.org/) to create a virtual environment.
- Install Conda on your machine
- Download this repository: `git clone https://github.com/gallorob/evocraft-life-project.git`
- Modify `evo_env_win.yml` or `evo_env_macos.yml` according to your OS, setting the `Prefix` variable to your local conda installation path
  - You can also change the environment name if it clashes with a preexisting one
- From within the repository folder, create the Conda environment: `conda env create -f evo_env_win.yml` or `conda env create -f evo_env_macos.yml`
- You can then activate the environment using `conda activate evo`
- Download the [EvoCraft API](https://github.com/real-itu/Evocraft-py) repository (you shouldn't need to install any additional dependency) and follow their installation instructions.
  - Optional: test if everything works by running their `example.py`
- Download the [PyTorch Neat](https://github.com/gallorob/PyTorch-NEAT) repository (you shouldn't need to install any additional dependency) and follow their installation instructions.
  - Optional: test if everything works by running their examples in the `examples` folder
- Copy `minecraft_pb2.py` and `minecraft_pb2_grpc.py` from `Evocraft-py` in the main `evocraft-life-project` directory.
- Copy `pytorch_neat` from `PyTorch-NEAT` in the main `evocraft-life-project` directory.

## Running experiments
- Make sure you are running the EvoCraft server. You may launch it issuing
  
  ```bash
  sh run_server.sh
  ```
- Modify the `experiment.cfg` and `neat.cfg` as you wish. The configuration presented here are used to evolve interesting artifacts (such as: statues), but your imagination (and resources) is the limit!
- Run the experiment. You may do this from scratch:
```bash
python3 main.py --seed={YOUR_SEED} --n={HOW_MANY_GENERATIONS} --from_scratch
```
or you can resume it with
```bash
python3 main.py --seed={YOUR_SEED} --n={HOW_MANY_GENERATIONS} --net={NETWORK_TIMESTEP_ID} --resume
```
- While the experiment is running, you will have to rate the structures you see appear in the Minecraft world. To do this, simply enter the number of the artifact in the command line when prompted.
  (remember: the numbers go from 1 to N, starting from the structure with the arrow next to it).

After the experiment is terminated, you can see the Fitness Estimator training curves using Tensorboard ETC...  