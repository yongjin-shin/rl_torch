from .ddpg import DDPG

REGISTRY = {}
REGISTRY["DDPG"] = DDPG
REGISTRY["MADDPG"] = DDPG
