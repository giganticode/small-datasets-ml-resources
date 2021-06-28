import torch
import torch.nn.functional as F
import math
import json

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

from dl4se.transformers.experiment import Experiment
import dl4se.transformers.util as tu
from dl4se import util
from dl4se.datasets.senti4sd import Dataset as SentiDataset, default_config
from dl4se.config.senti4sd import get_config


def parse_args(parser):
    parser.add_argument("--interp_out_file", default=None,
                        type=str, required=False)

    parser.add_argument("--jira", action='store_true', default=False)
    parser.add_argument("--app_reviews", action='store_true', default=False)
    parser.add_argument("--sentidata_so", action='store_true', default=False)

def main(config, results):
    pd.set_option('display.max_rows', None)

    model_config = tu.load_model_config(config)
    tokenizer = tu.load_tokenizer(config, model_config)

    ds = SentiDataset(config, tokenizer)
    test_dataloader = ds.get_test_dataloader()

    model = tu.load_model(config, model_config)
    model.to(config.device)
    util.set_seed(config)

    train_dataloader, valid_dataloader = ds.get_train_valid_dataloaders()
    test_dataloader = ds.get_test_dataloader()

    test_dataloaders = {'test': ds.get_test_dataloader()}

    if config.jira:
        test_dataloaders['JIRA'] = (ds.get_jira_dataloader(), dict(pred_label_ids_func=ds.neutral_to_negative))

    if config.app_reviews:
        test_dataloaders['AppReviews'] = ds.get_app_reviews_dataloader()

    if config.sentidata_so:
        test_dataloaders['StackOverflow (SentiData)'] = ds.get_stack_overflow_dataloader()

    experiment = Experiment(config, model, tokenizer, label_names=ds.label_names, results=results)
    global_step, tr_loss = experiment.train(train_dataloader, valid_dataloader=valid_dataloader, test_dataloader=test_dataloaders)

    # interp_df = experiment.interpret(test_dataloader, ds.test_df, label_names=ds.label_names)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(interp_df)

    # if config.interp_out_file:
    #     interp_df.to_csv(config.interp_out_file, index=False)

    return experiment.results

config = get_config(parse_args)
util.run_with_seeds(config, main)