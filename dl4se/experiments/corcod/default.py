import torch
import torch.nn.functional as F
import math
import json
import itertools

from dl4se.transformers.experiment import Experiment
import dl4se.transformers.util as tu
from dl4se import util

from dl4se.config.corcod import get_config
from dl4se.datasets.corcod import Dataset

def parse_args(parser):
    pass

def main(config, results):

    model_config = tu.load_model_config(config)
    tokenizer = tu.load_tokenizer(config, model_config)

    ds = Dataset(config, tokenizer)
    label_names = ds.label_names

    for i, (train_dataloader, (valid_dataloader, valid_df)) in enumerate(ds.get_train_valid_dataloaders(include_valid_df=True)):
        print(f"------------------ BEGIN ITER {i} -----------------------")
        # need to reload original model config, to avoid vocabulary size mismatch
        # caused by custom tokens
        model_config = tu.load_model_config(config)
        model = tu.load_model(config, model_config)
        model.to(config.device)
        util.set_seed(config)

        run_name = f"fold{i}"

        experiment = Experiment(config, model, tokenizer, label_names=label_names, run_name=run_name, results=results)
        global_step, tr_loss = experiment.train(
            train_dataloader, valid_dataloader=valid_dataloader)

        results = experiment.results
        # experiment.evaluate('valid', valid_dataloader)
        print(f"================== DONE ITER {i} =======================\n\n")

    return results

config = get_config(parse_args)
util.run_with_seeds(config, main)
