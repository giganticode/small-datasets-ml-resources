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

from dl4se.datasets.satd import Dataset as TDDataset

from dl4se.config.satd import get_config
from dl4se.logging import logger

def parse_args(parser):
    parser.add_argument("--single_project", default=None,
                        type=str, required=False)

    parser.add_argument("--interp_out_file", default=None,
                        type=str, required=False)


def main(config, results):

    logger.warning('Unclassified threshold: %s', config.self_train_thresh)

    ds = TDDataset(config, binary=True,
                   self_train_thresh=config.self_train_thresh,
                   keyword_masking_frac=config.keyword_masking_frac)

    model_config = tu.load_model_config(config)
    tokenizer = tu.load_tokenizer(config, model_config)
    label_names = ds.label_names

    #project_name = 'emf-2.4.1'
    project_name = config.single_project

    iter_obj = [(project_name, *ds.get_train_valid_dataloaders(tokenizer, project_name, include_valid_df=True))
                ] if project_name else ds.get_fold_dataloaders(tokenizer, include_valid_df=True)

    interp_out_file = Path(config.interp_out_file) if config.interp_out_file else None

    # for train_dataloader, valid_dataloader in [ds.get_train_valid_dataloaders(tokenizer, project_name)]:
    for project_name, train_dataloader, (valid_dataloader, valid_df) in iter_obj:
        print(
            f"------------------ BEGIN PROJECT {project_name} -----------------------")

        model = tu.load_model(config, model_config)
        model.to(config.device)
        util.set_seed(config)

        experiment = Experiment(
            config, model, tokenizer, label_names=label_names, run_name=project_name, results=results)
        global_step, tr_loss = experiment.train(
            train_dataloader, valid_dataloader=valid_dataloader)

        if interp_out_file:
            interp_df = experiment.interpret(valid_dataloader, valid_df)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(interp_df)
            interp_df.to_csv(interp_out_file.with_name(f"{project_name}_{interp_out_file.name}"), index=False)

        results = experiment.results

        # experiment.evaluate('valid', valid_dataloader)

        print(
            f"================== DONE PROJECT {project_name} =======================\n\n")
    return results	

config = get_config(parse_args)
util.run_with_seeds(config, main)
