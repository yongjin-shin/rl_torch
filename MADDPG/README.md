# MADDPG
The original [paper](https://arxiv.org/pdf/1706.02275.pdf) is openai's "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments". The authors open their [[1]code](https://github.com/openai/maddpg), however it seems that their code cannot reproduce the results of the paper well so that I just tried to write my own code with pytorch. The training was not trivial, thus this code is highly affected by MADDPG [[2]code](https://github.com/shariqiqbal2810/maddpg-pytorch) from [MAAC](https://arxiv.org/pdf/1810.02912.pdf)'s author.



The results is nearly similar with that of [2]code. Under the cooperative setting(simple_spread), my code does better, on the other hand, the competitive setting(simple_tag), [2]code's result is slightly better.



### Environments

```
python==3.6.10

torch==1.4.0

gym==0.14.0

git clone https://github.com/shariqiqbal2810/multiagent-particle-envs.git
```

[2]code's author made changes in the original [MPE](https://github.com/openai/multiagent-particle-envs). MADDPG only experimented under the discrete settings, I guess. Thus, [2]code's author modify the original code to let the user choose the environment setting, whether discrete or continuous. Some other changes could be checked from his github [repository](https://github.com/shariqiqbal2810/multiagent-particle-envs).



### Run

```
python main.py with args.scenario="scenario_name"
```

You can check some hyperparameter settings from ```multi_run.sh```.

Since I used ```sacred``` and ```omniboard```, you can change hyperparameter with using ``with`` syntax.



### Results

Note that unlike mujoco environments, batch normalization is critical issue during training! Also reward normalization is essential.

Every 5000 steps, 10 episodes of evaluation was preceded and recorded. The followings were from these evaluation results.



**Cooperative navigation (simple_spread)**

![spread](https://github.com/yongjin-shin/rl_torch/blob/master/MADDPG/assets/spread.gif)

![spread_graph](https://github.com/yongjin-shin/rl_torch/blob/master/MADDPG/assets/spread.png)



**Predator Prey (simple_tag)**

![tag](https://github.com/yongjin-shin/rl_torch/blob/master/MADDPG/assets/tag.gif)

![tag_graph](https://github.com/yongjin-shin/rl_torch/blob/master/MADDPG/assets/tag.png)

