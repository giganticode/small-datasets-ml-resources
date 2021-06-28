import torch
import torch.nn.functional as F
import math
import json
import itertools

from dl4se.transformers.experiment import Experiment
import dl4se.transformers.util as tu
from dl4se import util

from dl4se.config.coherence import get_config
from dl4se.datasets.coherence import Dataset

def parse_args(parser):
    parser.add_argument("--dataset", default='corazza',
                        type=str, required=False)

def main(config, results):

    model_config = tu.load_model_config(config)
    tokenizer = tu.load_tokenizer(config, model_config)

    ds = Dataset(config, tokenizer)
    label_names = ds.label_names

    train_dataloader, valid_dataloader, test_dataloader = getattr(ds, f"get_{config.dataset}_train_valid_test_dataloaders")()

    model = tu.load_model(config, model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(config.device)
    util.set_seed(config)

    experiment = Experiment(config, model, tokenizer, label_names=label_names, results=results)
    global_step, tr_loss = experiment.train(
         train_dataloader, valid_dataloader=valid_dataloader, test_dataloader=test_dataloader)

    results = experiment.results
    return results

config = get_config(parse_args)
util.run_with_seeds(config, main)
