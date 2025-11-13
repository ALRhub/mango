# 🥭 MaNGO: Adaptable Graph Network Simulators via Meta-Learning

This repository will host the official code release for our **NeurIPS 2025** paper:

> **[MaNGO: Adaptable Graph Network Simulators via Meta-Learning](https://arxiv.org/abs/2510.05874)**  
> Philipp Dahlinger, Tai Hoang, Denis Blessing, Niklas Freymuth, Gerhard Neumann
> Karlsruhe Institute of Technology (KIT)

Check out our [project page](https://alrhub.github.io/mango/) for an overview and visualizations of all tasks!

---

### Overview

MaNGO introduces a **Meta Neural Graph Operator** that enables **Graph Network Simulators** to adapt across different physical systems.  
The method combines **meta-learning** with a **neural operator based architecture**, allowing the simulator to generalize to unseen material properties and predict full trajectories efficiently and stably.

---

### Installation
We use `uv` as our environment manager. To set up the environment, run:

```bash
# Create a virtual environment (recommended)
uv venv
uv sync
```
For developing access, run:
```bash
uv pip install -e .
```

---

### Usage
If you just want to have a look at the decoder code as a baseline for your own experiments, check it out in `src/mango/simulator/ml_decoder/mango_decoder.py`.

For training the full MaNGO model on the datasets, you can use the training script `train.py`. You need to provide a hydra config file. As an example, you can run

```bash
uv python train.py +experiment/final_exp/cnn_deepset_mango=dp_easy_v5 +platform=local_multirun
```

For that to work you need to download the dataset here: [Dataset Download Link](https://zenodo.org/records/17287535).
Put the hdf5 files into a folder `../datasets/mango/` relative to the root of this repository (or update the path in the dataset configs in `configs/dataset/`)
### Contact

For questions, please contact:  
**Philipp Dahlinger** – [philipp.dahlinger@kit.edu](mailto:philipp.dahlinger@kit.edu)
