import yaml
import random
import numpy as np
import torch
from argparse import Namespace
from sacred import Experiment
from sacred.observers import MongoObserver
from utils.misc import cuda_check
from modules.runner import Simple_Runner
from wrappers.env_wrapper import Envrionment


ex = Experiment('ddpg')
ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='sacred'))
f = open("config/config.yaml")
config_dict = yaml.load(f)


@ex.config
def config():
    args = config_dict


@ex.automain
def main(args):
    args = Namespace(**args)
    train_env = Envrionment(args, discrete_action=args.discrete_action)
    eval_env = Envrionment(args, discrete_action=args.discrete_action)
    args = train_env.args_update()
    assert args.atypes == eval_env.get_agent_types()

    # for the reproducibility
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    train_env.seed(int(args.seed))
    eval_env.seed(int(args.seed)+1001)
    args.device = cuda_check(args)

    # Training
    runner = Simple_Runner(args, ex)
    runner.train(env=train_env, eval_env=eval_env)
    print("Training has been finished!")

    # Evaluation
    # runner.path = './result/simple_spread/2020-01-28-13:48'
    runner.model_load()
    runner.args.model_saving = False
    runner.eval(eval_env, render_saving=True)
    print("Evaluation has been finished!")
