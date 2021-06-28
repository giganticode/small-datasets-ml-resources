import torch
import torch.nn.functional as F
import math
import json
import itertools

from dl4se.transformers.experiment import Experiment
import dl4se.transformers.util as tu
from dl4se import util

from dl4se.config.coherence import get_config
from dl4se.datasets.coherence import ShuffleDataset as Dataset

def parse_args(parser):
    pass

def main(config, results):
    model_config = tu.load_model_config(config)
    tokenizer = tu.load_tokenizer(config, model_config)

    ds = Dataset(config, tokenizer)
    label_names = ds.label_names

    train_dataloader, valid_dataloader = ds.get_train_valid_dataloaders()

    model = tu.load_model(config, model_config)
    model.to(config.device)
    util.set_seed(config)

    experiment = Experiment(config, model, tokenizer, label_names=label_names, results=results)
    global_step, tr_loss = experiment.train(train_dataloader, valid_dataloader=valid_dataloader)
    results = experiment.results

    experiment.save_model(util.models_path('comment_code_shuffle'))
    
    return results

config = get_config(parse_args)

config.logging_steps = 1000
config.train_epochs = 10

util.run_with_seeds(config, main)
