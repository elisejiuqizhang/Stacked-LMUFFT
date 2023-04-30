# Stacked LMUFFT

Stacking parallelized Legendre Memory Unit (LMUFFT) for sequence modeling.

## Environment

Check out the `environment.yml`, `notes_on_env.txt` for details. For conda users, I think it would be relatively easy to execute `create_env.sh` from terminal.

## Datasets

Store the .csv files under the `data` folder. Some sample datasets can be acquired through the links: [ETT](https://github.com/zhouhaoyi/ETDataset) from the [Informer paper](https://arxiv.org/abs/2012.07436), [Weather](https://drive.google.com/drive/folders/1Xz84ci5YKWL6O2I-58ZsVe42lYIfqui1?usp=share_link) from the [Autoformer paper](https://arxiv.org/abs/2106.13008).

For using your own customized dataset, you might need to modify the dataloaders (under the `data_loaders` folder) as you see fit.

## Run experiments

Check `main.py` for all the arguments and execute it from terminal. Example: 
```
python main.py --model LMUFFT --num_layers 2 --dataset ETTh1
```