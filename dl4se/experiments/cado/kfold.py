import torch
import torch.nn.functional as F
import math
import json
import itertools

from dl4se.transformers.experiment import Experiment
import dl4se.transformers.util as tu
from dl4se import util

from dl4se.config.cado import get_config
from dl4se.datasets.cado import Dataset as CadoDataset

def parse_args(parser):
    pass

def main():
    config = get_config(parse_args)
    util.set_seed(config)

    with config:
        config.logging_steps = 50
        config.train_epochs = 5

    # config.train_head_only = True

    print("model is now", config.model_path)
    ds = CadoDataset(config)
    label_names = ds.label_names

    model_config = tu.load_model_config(config)
    tokenizer = tu.load_tokenizer(config, model_config)

    f1s = []
    results = None

    test_dataloader = ds.get_test_dataloader(tokenizer)

    for i, (train_dataloader, (valid_dataloader, valid_df)) in enumerate(ds.get_all_train_valid_dataloaders(tokenizer, include_valid_df=True)):
        print(f"------------------ BEGIN ITER {i} -----------------------")
        model = tu.load_model(config, model_config)
        model.to(config.device)
        util.set_seed(config)

        run_name = f"{config.single_class if config.single_class else 'multi'}_{i}"

        experiment = Experiment(config, model, tokenizer, label_names=label_names, run_name=run_name, results=results)
        global_step, tr_loss = experiment.train(
            train_dataloader, valid_dataloader=valid_dataloader, test_dataloader=test_dataloader)

        results = experiment.results
        # experiment.evaluate('valid', valid_dataloader)
        print(f"================== DONE ITER {i} =======================\n\n")

main()
