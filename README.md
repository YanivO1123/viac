# Installation instructions with conda:
Unzip the files into a new directory (named viac in this example).
Execute the following:
```
conda env create -f path_to_viac_dir/viac/viac_env.yml
conda activate viac
pip install gymnasium==0.29.0 --no-dependencies
pip install git+https://github.com/imgeorgiev/dmc2gymnasium.git
```

# Instructions to running the code

## (VI-)TD3:
TD3:
```
python /path_to_viac/viac/vi_td3_continuous_action.py --env-id=hopper-hop --total_timesteps=3000000 --learning_starts=10000
```
VI-TD3 with expectile loss:
```
python /path_to_viac/viac/vi_td3_continuous_action.py --env-id=hopper-hop --total_timesteps=3000000 --learning_starts=10000 --value_improvement_operator="expectile"
```
VI-TD3 with policy gradient value improvement:
```
python /path_to_viac/viac/vi_td3_continuous_action.py --env-id=hopper-hop --total_timesteps=3000000 --learning_starts=10000 --value_improvement_operator="policy_gradient" --number_of_vi_gradient_updates=20
```
VI-TD3 with sampling-based argmax:
```
python /path_to_viac/viac/vi_td3_continuous_action.py --env-id=hopper-hop --total_timesteps=3000000 --learning_starts=10000 --value_improvement_operator="sampled_argmax" --n=128
```

## (VI-)SAC:
SAC:
```
python /path_to_viac/viac/vi_sac_continuous_action.py --env-id=hopper-hop --total_timesteps=3000000 --learning_starts=10000
```
VI-SAC with expectile loss:
```
python /path_to_viac/viac/vi_sac_continuous_action.py --env-id=hopper-hop --total_timesteps=3000000 --learning_starts=10000 --value_improvement_operator="expectile"
```


## (VI-)TD7:
SAC:
```
python /path_to_viac/viac/vi_td7_continuous_action.py --env-id=hopper-hop --total_timesteps=3000000 --learning_starts=10000
```
VI-SAC with expectile loss:
```
python /path_to_viac/viac/vi_td7_continuous_action.py --env-id=hopper-hop --total_timesteps=3000000 --learning_starts=10000 --value_improvement_operator="expectile"
```


All experiments can be tracked in WANDB by adding:
```
--wandb_project_name=project_name --track
```