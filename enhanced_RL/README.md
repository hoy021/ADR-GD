## Enhancing TD3 training

Install the packages [stable baseline3](https://github.com/DLR-RM/stable-baselines3/tree/master) and [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

Replace the following files in the installed packages, with the ones in this repo:
- `stable-baselines3/stable_baselines3/td3/td3.py`
- `stable-baselines3/stable_baselines3/common/off_policy_algorithm.py`
- `rl-baselines3-zoo/blob/master/hyperparams/td3.yml`

Start the training of modified TD3, in the root folder of `rl-baselines3-zoo`:
```
python train.py --algo td3 --env Ant-v3/HalfCheetah-v3/Walker2d-v2
```
