import torch
import torch.nn.functional as F
import math
import json
import itertools

import pandas as pd

from dl4se.transformers.experiment import Experiment
import dl4se.transformers.util as tu
from dl4se import util

from dl4se.config.satd import get_config
from dl4se.datasets.satd import Dataset as TDDataset


def main():
    config = get_config()
    with config:
        config.logging_steps = 400
        config.train_epochs = 2
        config.lr = 4e-5
        # config.lr = 1e-4
        config.model_type = 'roberta'
        config.model_path = util.models_path('StackOBERTflow-comments-small-v1')
        # config.train_head_only = True

    ds = TDDataset(config, binary=True)

    tokenizer = tu.load_tokenizer(config)
    model_cls = tu.get_model_cls(config)

    train_dataloader = ds.get_complete_train_dataloader(tokenizer)
    model = tu.load_model(config)
    model.to(config.device)
    util.set_seed(config)

    experiment = Experiment(config, model, tokenizer)
    global_step, tr_loss = experiment.train(train_dataloader)

    experiment.save_model(util.models_path('satd_complete_binary'))


main()
