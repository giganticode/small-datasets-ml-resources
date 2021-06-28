import dl4se.transformers.util as tu
from dl4se import util
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from dl4se.logging import logger

from sklearn.model_selection import train_test_split

from dl4se.datasets.util import preprocess_comment_soft

from itertools import permutations

class ShuffleDataset:
    LABEL_NAMES = ['random', 'right']

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.train_df = self.__preprocess_df(pd.read_csv(util.data_path("coherence", "shuffle", "train.csv")))
        self.valid_df = self.__preprocess_df(pd.read_csv(util.data_path("coherence", "shuffle", "valid.csv")))

    def __preprocess_df(self, df):

        # let the tokenizer join comment and code with a separator token
        df['preprocessed'] = '/* ' + df.comment + ' */\n' + df.code
        print(df.head(2))
        return df

    @property
    def label_names(self):
        return self.LABEL_NAMES

    def get_dataloader(self, df, bs, include_df=False, **kwargs):
        text_values = df.preprocessed
        label_ids = df['class'].values
        dataloader = tu.get_dataloader(self.config, self.tokenizer, text_values, label_ids, bs, **kwargs) 

        if include_df:
            return (dataloader, df)
        return dataloader        

    def get_train_valid_dataloaders(self, include_valid_df=False):
        return (self.get_dataloader(self.train_df, bs=self.config.train_bs),
                self.get_dataloader(self.valid_df, bs=self.config.eval_bs, include_df=include_valid_df))

class Dataset:
    LABEL_NAMES = ['non_coherent', 'coherent']

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.load_dfs()

    @property
    def n_splits(self):
        return 10

    @property
    def label_names(self):
        return self.LABEL_NAMES

    def get_dataloader(self, df, bs, include_df=False, **kwargs):

        if self.config.sep_token:
            text_values1 = df.preprocessed_comment.values
            text_values2 = df.preprocessed_code.values
        else:
            text_values1 = df.preprocessed.values
            text_values2 = None

        label_ids = df.label.values
        dataloader = tu.get_dataloader(self.config, self.tokenizer, text_values1, label_ids, bs, text_pair_values=text_values2, **kwargs) 

        if include_df:
            return (dataloader, df)

        return dataloader        

    def __preprocess_df(self, df, code_col, comment_col='comment'):

        # let the tokenizer join comment and code with a separator token
        if self.config.sep_token:
            df['preprocessed_comment'] = df[comment_col].map(preprocess_comment_soft)
            df['preprocessed_code'] = df[code_col]
            return df

        first, second = df[comment_col], df[code_col]
        if self.config.comment_last:
            first, second = second, first

        df['preprocessed'] = first + '\n' + second
        print(df.head(2))        
        return df

    def __preprocess(self, code):
        pass

    def load_dfs(self):
        self.corazza_df = self.__preprocess_df(pd.read_csv(util.data_path("coherence", "corazza.csv")), "method")
        self.wang_df = self.__preprocess_df(pd.read_csv(util.data_path("coherence", "wang.csv")), "code")

    def _get_train_valid_test_dataloaders(self, df, include_valid_df, balance=False):
        train_df, test_df = train_test_split(df, test_size=0.25, random_state=self.config.seed)
        train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=self.config.seed)

        return (self.get_dataloader(train_df, bs=self.config.train_bs, balance=balance),
                self.get_dataloader(valid_df, bs=self.config.eval_bs, include_df=include_valid_df),
                self.get_dataloader(test_df, bs=self.config.eval_bs))

    def get_corazza_train_valid_test_dataloaders(self, include_valid_df=False):
        return self._get_train_valid_test_dataloaders(self.corazza_df, include_valid_df, balance=self.config.balance)

    def get_wang_train_valid_test_dataloaders(self, include_valid_df=False):
        return self._get_train_valid_test_dataloaders(self.wang_df, include_valid_df, balance=True)