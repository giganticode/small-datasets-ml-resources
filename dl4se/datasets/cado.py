import dl4se.transformers.util as tu
from dl4se import util
import pandas as pd

from functools import cached_property

from sklearn.model_selection import KFold

from dl4se.datasets.util import preprocess_comment

class Dataset(object):

    LABEL_COLS = ["functionality", "concept", "directives", "purpose",
                  "quality", "control", "structure", "patterns", "codeExamples", "environment", "reference", "nonInformation"]

    LABELS = LABEL_COLS #+ ['none']

    def __init__(self, config):
        self.config = config

        self.load_df()

    @property
    def n_splits(self):
        return 10

    @property
    def label_names(self):
        if self.config.single_class:
            return [f'Not {self.config.single_class}', self.config.single_class]
        else:
            return Dataset.LABELS

    def get_dataloader(self, tokenizer, df, bs, include_df=False, **kwargs):
        text_values = df.preprocessed.values
        single_class = self.config.single_class

        if self.config.single_class:
            label_ids = df[single_class].to_numpy()
        else:
            label_ids = df[Dataset.LABELS].to_numpy()

        dataloader = tu.get_dataloader(self.config, tokenizer, text_values, label_ids, bs, **kwargs) 

        if include_df:
            return (dataloader, df)

        return dataloader

    def load_df(self):
        df = pd.read_csv(util.data_path(
            "cado", "dataset.csv"), delimiter=',')
        self.__add_none_label(df)

        self.orig_df = df
        self.df = self.preprocess_df(df)
        self.test_df = None

    def load_test_df(self):
        if not self.test_df is None:
            return

        df = pd.read_csv(util.data_path(
            "cado", "python_stdlib.csv"), delimiter=',')
        self.__add_none_label(df)

        self.test_df = self.preprocess_df(df, col='text')

    def __add_none_label(self, df):
        pass
        #df['none'] = (df[Dataset.LABEL_COLS] == 0).all(axis=1).astype(int)


    def preprocess(self, text):
        text = text.replace('Additional online Documentation: Syntax:', '')
        text = text.replace('Additional online Documentation: Summary:', '')
        text = preprocess_comment(text)

        return text


    def preprocess_df(self, df, col='documentText'):
        df = df.assign(preprocessed = df[col].map(self.preprocess))
        df.dropna(inplace=True)

        return df

    def get_test_dataloader(self, tokenizer):
        self.load_test_df()
        return self.get_dataloader(tokenizer, self.test_df, bs=self.config.eval_bs)

    def get_all_train_valid_dataloaders(self, tokenizer, single_class=None, include_valid_df=False):
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.config.seed)
        splits = kfold.split(self.df)

        for (train_idxs, test_idxs) in splits:
            train_df = self.df.iloc[train_idxs]
            valid_df = self.df.iloc[test_idxs]

            yield (self.get_dataloader(tokenizer, train_df, bs=self.config.train_bs, balance=False),
                   self.get_dataloader(tokenizer, valid_df, bs=self.config.eval_bs, include_df=include_valid_df))

