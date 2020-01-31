import gym
import yaml
import numpy as np
import torch
from argparse import Namespace
from sacred import Experiment
from sacred.observers import MongoObserver
from modules.runner import Simple_Runner
from modules.utils import args_update


ex = Experiment('ddpg')
ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='sacred'))
f = open("config/config.yaml")
config_dict = yaml.load(f)


@ex.config
def config():
    args = config_dict


@ex.automain
def main(args):
    train_env = gym.make(args['scenario'])
    eval_env = gym.make(args['scenario'])
    args = Namespace(**args)

    # for the reproducibility
    np.random.seed(int(args.seed))
    train_env.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = args_update(args, train_env)
    args.device = device
    for k in sorted(vars(args).keys()):
        print("{}: {}".format(k, vars(args)[k]))

    runner = Simple_Runner(args, train_env, eval_env, ex)
    runner.train()
