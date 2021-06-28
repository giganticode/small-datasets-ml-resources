import torch
import torch.nn.functional as F
import math
import json
import itertools

from dl4se.transformers.experiment import Experiment
import dl4se.transformers.util as tu
from dl4se import util

from dl4se.config.ar_miner import get_config
from dl4se.datasets.ar_miner import Dataset

from dl4se.logging import logger

def parse_args(parser):
    pass

def main(config, results):
    model_config = tu.load_model_config(config)
    tokenizer = tu.load_tokenizer(config, model_config)

    ds = Dataset(config, tokenizer)
    label_names = ds.label_names

    train_dataloader, valid_dataloader = ds.get_train_valid_dataloaders()
    test_dataloader = ds.get_test_dataloader()

    # with config:
    #     config.max_steps=100

    model = tu.load_model(config, model_config)
    model.to(config.device)
    util.set_seed(config)

    experiment = Experiment(config, model, tokenizer, label_names=label_names, results=results)
    global_step, tr_loss = experiment.train(train_dataloader, valid_dataloader=valid_dataloader, test_dataloader=test_dataloader)

    results = experiment.results
    
    return results

config = get_config(parse_args)
util.run_with_seeds(config, main)