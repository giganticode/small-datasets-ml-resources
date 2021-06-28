import torch
import torch.nn.functional as F
import math
import json
import itertools
import pandas as pd
from pathlib import Path

from dl4se.transformers.experiment import Experiment
import dl4se.transformers.util as tu
from dl4se import util

from dl4se.config.review_classification import get_config
from dl4se.datasets.review_classification import Dataset, ClapDataset

def parse_args(parser):
    parser.add_argument("--interp_out_file", default=None,
                        type=str, required=False)

def main(config, results):
    model_config = tu.load_model_config(config)
    tokenizer = tu.load_tokenizer(config, model_config)

    if config.clap:
        ds = ClapDataset(config, tokenizer)
    else:
        ds = Dataset(config, tokenizer)

    label_names = ds.label_names
    model_config = tu.load_model_config(config)

    interp_out_file = Path(config.interp_out_file) if config.interp_out_file else None

    for i, (train_dataloader, (valid_dataloader, valid_df)) in enumerate(ds.get_train_valid_dataloaders(include_valid_df=True)):
        print(f"------------------ BEGIN ITER {i} -----------------------")
        model = tu.load_model(config, model_config)
        model.to(config.device)
        util.set_seed(config)

        run_name = f"fold{i}"

        experiment = Experiment(config, model, tokenizer, label_names=label_names, run_name=run_name, results=results)
        global_step, tr_loss = experiment.train(
            train_dataloader, valid_dataloader=valid_dataloader)


        if interp_out_file:
            interp_df = experiment.interpret(valid_dataloader, valid_df, label_names=label_names)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(interp_df)
            interp_df.to_csv(interp_out_file.with_name(f"{interp_out_file.name}_iter{i}"), index=False)

        results = experiment.results
        # experiment.evaluate('valid', valid_dataloader)
        print(f"================== DONE ITER {i} =======================\n\n")

    return results

config = get_config(parse_args)
util.run_with_seeds(config, main)
