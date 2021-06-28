import torch
import torch.nn.functional as F
import math
import json
import itertools

import pandas as pd

import numpy as np

from dl4se.transformers.experiment import Experiment
import dl4se.transformers.util as tu
from dl4se import util

from dl4se.config.satd import get_config
from dl4se.datasets.satd import Dataset as TDDataset

from dl4se.active_learn.acquisition_funcs import *


def main():
    config = get_config()
    with config:
        config.logging_steps = 400
        config.train_epochs = 2
        config.lr = 4e-5
        # config.lr = 1e-4
        config.model_type = 'roberta'
        config.model_path = util.models_path('satd_complete_binary')
        # config.train_head_only = True

    tokenizer = tu.load_tokenizer(config)
    model_cls = tu.get_model_cls(config)

    df = pd.read_csv(util.data_path('satd', 'unclassified.csv'))
    # df = pd.read_csv(util.data_path('satd', 'dataset.csv'))
    df.dropna(inplace=True)
    # df.rename(columns={'classification': 'orig_classification'}, inplace=True)

    print(df.dtypes)

    print(df.head())

    df['preprocessed'] = df.commenttext.map(TDDataset.preprocess)
    df.dropna(inplace=True)
    # df = df.head(100)
    preprocessed = df.preprocessed.values
    dummy_labels = np.zeros(preprocessed.shape[0])
    dataloader = tu.get_dataloader(config, tokenizer, preprocessed, dummy_labels, bs=128, shuffle=False)

    model = tu.load_model(config)
    model.to(config.device)
    util.set_seed(config)

    experiment = Experiment(config, model, tokenizer)

    preds = experiment.predict(dataloader)
    preds = torch.from_numpy(preds)
    probs = F.softmax(preds, dim=1)
    uncertainty = least_conf(probs).numpy()
    labels = np.argmax(preds, axis=1)

    df['uncertainty'] = uncertainty
    df['probs0'] = probs[:, 0].numpy()
    df['probs1'] = probs[:, 1].numpy()
    df['classification'] = labels
    df.drop('preprocessed', axis='columns', inplace=True)
    
    label_name_map = {i:l for i, l in enumerate(TDDataset.BINARY_LABEL_NAMES)}
    print(label_name_map)


    # convert_label = {'DEFECT': 1, 'DESIGN': 1,
    #                  'IMPLEMENTATION': 1, 'TEST': 1,
    #                  'WITHOUT_CLASSIFICATION': 0, 'DOCUMENTATION': 1}
    # df['correct'] = (df.orig_classification.map(convert_label) == df.classification)
    # print(df.correct.value_counts(normalize=True))

    df.classification = df.classification.map(label_name_map)
    df.to_csv(util.data_path('satd', 'unclassified_evaled.csv'), index=False)

    tech_debt_df = df[df.classification == 'TECHNICAL_DEBT']
    print(tech_debt_df.shape)
    tech_debt_df.to_csv(util.data_path('satd', 'unclassified_pos.csv'), index=False)




main()
