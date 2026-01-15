# Value Improved Actor Critic Algorithms
This repository contains the code used to run the experiments in the Value Improved Actor Critic Algorithms (VIAC) paper.
VIAC introduces the notion of *value improvement*, the inclusion of *policy improvement* inside the evaluation step, to address the gap between how much we may want to improve the policy, and how much gradient based updates enable improving the policy in RL algorithms. In actor–critic methods specifically, as well as more generally in policy- and value-based algorithms that use deep neural networks for function approximation.

See the paper for more information.

## Installation:
```
conda env create -f path_to_viac_dir/viac/viac_env.yml
conda activate viac
pip install gymnasium==0.29.0 --no-dependencies
pip install git+https://github.com/imgeorgiev/dmc2gymnasium.git
```

## Example usage:
VI-TD3 with expectile loss:
```
python /path_to_viac/viac/vi_td3_continuous_action.py --env-id=hopper-hop --total_timesteps=3000000 --learning_starts=10000 --value_improvement_operator="expectile"
```
VI-SAC with expectile loss:
```
python /path_to_viac/viac/vi_sac_continuous_action.py --env-id=hopper-hop --total_timesteps=3000000 --learning_starts=10000 --value_improvement_operator="expectile"
```
VI-TD7 with expectile loss:
```
python /path_to_viac/viac/vi_td7_continuous_action.py --env-id=hopper-hop --total_timesteps=3000000 --learning_starts=10000 --value_improvement_operator="expectile"
```

All experiments can be tracked in Weights & Biases (wandb) by adding:
```
--wandb_project_name=project_name --track
```

See the ```Args``` class in each file for description of other command-line arguments.

## Acknowledgements

This repository is based on a fork of the [CleanRL](https://github.com/vwxyzjn/cleanrl) repository, with value-improvement extensions to their implementations of [TD3](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) and [SAC](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py).

The implementation of VI-TD7 is based on the [official repository](https://github.com/sfujim/TD7) for the [TD7 (For SALE: State-Action Representation Learning for Deep Reinforcement Learning) paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/c20ac0df6c213db6d3a930fe9c7296c8-Paper-Conference.pdf).

## Citation:
Please cite us as:
```
@inproceedings{viac,
  title     = {{Value Improved Actor Critic Algorithms}},
  author    = {Oren, Yaniv and Zanger, Moritz A. and Van der Vaart, Pascal R. and {\c{C}}elikok, Mustafa Mert and Spaan, Matthijs T. J. and Boehmer, Wendelin},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025},
}

```