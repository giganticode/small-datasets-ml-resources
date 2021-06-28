import os
import numpy as np
import torch
import logging
import random
import json
from pathlib import Path

from dl4se.logging import logger

class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


__root_path = Path(__file__).parent.parent
__data_path = __root_path / 'data'
__model_path = __root_path / 'models'


def root_path(*filename):
    return str(__root_path.joinpath(*filename))


def models_path(*filename):
    return str(__model_path.joinpath(*filename))


def data_path(*filename):
    return str(__data_path.joinpath(*filename))


def set_seed(config_or_seed):
    if isinstance(config_or_seed, int):
        seed = config_or_seed
    else:
        seed = config_or_seed.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.warn(f'Setting seed to {seed}')
    #if config.n_gpu > 0:
    torch.cuda.manual_seed_all(seed)


def run_with_seeds(config, func, *args, **kwargs):
    print(config, func)
    seeds = config.seeds
    results = None

    if not seeds: seeds = [config.seed]

    for seed in seeds:
        config.seed = seed
        set_seed(config)
        results = func(config, results, *args, **kwargs)
        if results is None:
            raise ValueError('function should return result object')