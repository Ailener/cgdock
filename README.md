<h1 align="center">
CGDock: Curvature-Aware Geometric Flow Framework for Protein-Ligand Docking
</h1>

## Overview
This repository contains the source code of CGDock. If you have questions, don't hesitate to open an issue or ask me via <jialy5@mail2.sysu.edu.cn>. I am happy to hear from you!

## Setup Environment
```shell
conda create --name CGDock python=3.8
conda activate CGDock
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.15%2Bpt112cu113-cp38-cp38-linux_x86_64.whl 
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/pyg_lib-0.2.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install torch-geometric==2.4.0 torchdrug==0.1.2 torchmetrics==0.10.2 tqdm mlcrate pyarrow accelerate Bio lmdb fair-esm tensorboard fair-esm rdkit-pypi==2021.03.4
conda install -c conda-forge openbabel 
```

## Data 
The PDBbind 2020 dataset can be download from http://www.pdbbind.org.cn. The training data we used comes from FAbind, which you can download via this [link](https://zenodo.org/records/11352521).

Before training or evaluation, please generate the ESM2 embedding vectors for the proteins based on the above preprocessed data.

```shell
python data_processing/generate_esm2_t33.py ${data_path}
```

Construct local curvature features and incorporate them into protein and ligand representations.

## Training mode
We provide pre-trained model weights, which you can download via [best_model]()

Evaluate the model using the following code.

```shell
python test.py
```

Retrain the model using the following command.

```shell
accelerate launch main.py
```
## About
### Acknowledegments
We appreciate [EquiBind](https://github.com/HannesStark/EquiBind), [TankBind](https://github.com/luwei0917/TankBind), [E3Bind](https://openreview.net/forum?id=sO1QiAftQFv), [DiffDock](https://github.com/gcorso/DiffDock), [FABind](https://github.com/QizhiPei/FABind/tree/main/FABind) and other related works for their open-sourced contributions.
