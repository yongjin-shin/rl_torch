# DDPG
The original [paper](https://arxiv.org/abs/1509.02971) is "Continuous Control with deep reinforcement learning". However this code is reproduced between the original paper and TD3's code [paper](https://arxiv.org/pdf/1802.09477.pdf) and their [code](https://github.com/sfujim/TD3). I didn't want to fully reproduce TD3's modified version of DDPG, however this code should be in between the original DDPG and the modified DDPG. The result is also in between two of them. 



### Environments

```
python==3.5.6

torch==1.3.1

gym==0.15.4
```

You can install all dependencies with ```requirements.txt```.



### Run

```
python main.py with --args.senario='scenario_name'
```

You can check some hyperparameter settings from ```run.sh```.

Since I used ```sacred``` and ```omniboard```, you can change hyperparameter with using ``with`` syntax.



### Results

Note that I didn't use batch normalization for the mujoco environments.

Every 5000 steps, 10 episodes of evaluation was preceded and recorded. The followings were from these evaluation results.



![Halfcheetah-v2](https://github.com/yongjin-shin/rl_torch/blob/master/DDPG/assets/half.png)

![Hopper-v2](https://github.com/yongjin-shin/rl_torch/blob/master/DDPG/assets/hopper.png)

![Ant-v2](https://github.com/yongjin-shin/rl_torch/blob/master/DDPG/assets/ant.png)

![Invertedpendulum-v2](https://github.com/yongjin-shin/rl_torch/blob/master/DDPG/assets/invertedpen.png)

![Walker-v2](https://github.com/yongjin-shin/rl_torch/blob/master/DDPG/assets/walker.png)

![Reacher-v2](https://github.com/yongjin-shin/rl_torch/blob/master/DDPG/assets/reacher.png)





