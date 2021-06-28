import dl4se.transformers.util as tu
from dl4se import util
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from dl4se.logging import logger

from sklearn.model_selection import train_test_split

class Dataset:
    LABEL_NAMES = ['non-informative', 'informative']

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.load_dfs()

    @property
    def label_names(self):
        return self.LABEL_NAMES

    def get_dataloader(self, df, bs, include_df=False, **kwargs):
        text_values = df.text.values
        label_ids = df.informative.astype(int).values
        dataloader = tu.get_dataloader(self.config, self.tokenizer, text_values, label_ids, bs, **kwargs) 

        if include_df:
            return (dataloader, df)

        return dataloader

    def __preprocess_df(self, df):
        return df

    def load_dfs(self):
        self.train_df = self.__preprocess_df(pd.read_csv(util.data_path("ar_miner", f"{self.config.dataset}_train.csv")))
        self.test_df = self.__preprocess_df(pd.read_csv(util.data_path("ar_miner", f"{self.config.dataset}_test.csv")))

    def get_test_dataloader(self):
        return self.get_dataloader(self.test_df, bs=self.config.eval_bs)

    def get_train_valid_dataloaders(self, include_valid_df=False):
        train_df, valid_df = train_test_split(self.train_df, test_size=0.15, random_state=self.config.seed)
        return (self.get_dataloader(train_df, bs=self.config.train_bs, balance=True),
                self.get_dataloader(valid_df, bs=self.config.eval_bs, include_df=include_valid_df))

