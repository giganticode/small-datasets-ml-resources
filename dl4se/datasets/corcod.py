import dl4se.transformers.util as tu
from dl4se import util
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from dl4se.logging import logger

class Dataset:
    LABEL_NAMES = ['1', 'logn', 'n', 'nlogn', 'n_square']

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.load_dfs()

    @property
    def n_splits(self):
        return 5

    @property
    def label_names(self):
        return self.LABEL_NAMES

    def get_dataloader(self, df, bs, include_df=False, **kwargs):
        text_values = df.code.values
        label_ids = df.complexity.map({label:idx for idx, label in enumerate(self.LABEL_NAMES)}).values

        if self.config.sep_token:
            # leave text section empty
            dataloader = tu.get_dataloader(self.config, self.tokenizer, [''] * len(text_values), label_ids, bs, text_pair_values=text_values, **kwargs) 
        else:            
            dataloader = tu.get_dataloader(self.config, self.tokenizer, text_values, label_ids, bs, **kwargs) 

        if include_df:
            return (dataloader, df)

        return dataloader

    def __preprocess_df(self, df):
        return df

    def load_dfs(self):
        self.df = self.__preprocess_df(pd.read_csv(util.data_path("corcod", "dataset.csv")))

    def get_train_valid_dataloaders(self, include_valid_df=False):
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.config.seed)
        splits = kfold.split(self.df)

        for (train_idxs, valid_idxs) in splits:
            train_df = self.df.iloc[train_idxs]
            valid_df = self.df.iloc[valid_idxs]

            yield (self.get_dataloader(train_df, bs=self.config.train_bs, balance=True),
                   self.get_dataloader(valid_df, bs=self.config.eval_bs, include_df=include_valid_df))

