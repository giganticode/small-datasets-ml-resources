import dl4se.transformers.util as tu
from dl4se import util
import pandas as pd

from functools import cached_property

import numpy as np
import re

from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from dl4se.datasets.util import preprocess_comment_soft, load_backtrans_dfs

from dl4se.util import logger

class Dataset:

    LABEL_NAMES = [
        'Rating',
        'UserExperience',
        'Bug',
        'Feature'
    ]

    LABEL_MAP = {label: i for i, label in enumerate(LABEL_NAMES)}

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.load_df()

    @property
    def n_splits(self):
        return 10

    @property
    def label_names(self):
        return self.LABEL_NAMES

    def get_dataloader(self, df, bs, backtrans_langs=[], include_df=False, **kwargs):
        text_values = df.preprocessed.values
        label_ids = df.label_id.astype(int).values
        
        if backtrans_langs:
            logger.info("Text values array shape before augmentation: %s", text_values.shape)
            logger.info("Label ids array shape before augmentation: %s", label_ids.shape)

            for l in backtrans_langs:
                text_values = np.append(text_values, df[f'preprocessed_{l}'].values, axis=0)
            label_ids = np.tile(label_ids, len(backtrans_langs) + 1)

            logger.info("Text values array shape AFTER augmentation: %s", text_values.shape)
            logger.info("Label ids array shape AFTER augmentation: %s", label_ids.shape)

        dataloader = tu.get_dataloader(self.config, self.tokenizer, text_values, label_ids, bs, **kwargs) 

        if include_df:
            return (dataloader, df)

        return dataloader        

    def __preprocess_df(self, df):
        df = df[~df.comment.isna()]
        df['label_id'] = df.label.map(self.LABEL_MAP)

        def preprocess_cols(title_col, comment_col):
            col = (df.rating.astype(str) + ' ' + df[title_col].fillna(' ') + ' ' + df[comment_col].fillna(' ')).str.strip()
            col = col.str.replace(r'\s+', ' ')
            return col

        df['preprocessed'] = preprocess_cols('title', 'comment')
        all_backtrans_langs = list(set(self.config.train_backtrans_langs) | set(self.config.test_backtrans_langs))
        for l in all_backtrans_langs:
            df[f'preprocessed_{l}'] = preprocess_cols(f'title_{l}', f'comment_{l}')

        print(df.preprocessed.values[:4])

        if all_backtrans_langs:
            print(df.preprocessed_fr.values[:4])
            print(df.preprocessed_de.values[:4])

        return df

    def load_df(self):
        self.df = self.__preprocess_df(pd.read_csv(util.data_path("review_classification", "reviews_backtrans.csv")))

    def get_train_valid_dataloaders(self, include_valid_df=False):
        rs = ShuffleSplit(n_splits=self.n_splits, test_size=0.3, random_state=self.config.seed)
        splits = rs.split(self.df)

        for (train_idxs, valid_idxs) in splits:
            train_df = self.df.iloc[train_idxs]
            valid_df = self.df.iloc[valid_idxs]

            yield (self.get_dataloader(train_df, bs=self.config.train_bs, balance=True, backtrans_langs=self.config.train_backtrans_langs),
                   self.get_dataloader(valid_df, bs=self.config.eval_bs, shuffle=False, include_df=include_valid_df, backtrans_langs=self.config.test_backtrans_langs))



class ClapDataset:

    LABEL_NAMES = [
        'OTHER',      
        'BUG',
        'FEATURE',
        'PERFORMANCE',
        'USABILITY',
        'ENERGY',
        'SECURITY',
    ]

    LABEL_MAP = {label: i for i, label in enumerate(LABEL_NAMES)}

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.load_df()

    @property
    def n_splits(self):
        return 10

    @property
    def label_names(self):
        return self.LABEL_NAMES

    def get_dataloader(self, df, bs, backtrans_langs=[], include_df=False, **kwargs):
        text_values = df.preprocessed.values
        label_ids = df.label_id.astype(int).values
        
        if backtrans_langs:
            logger.info("Text values array shape before augmentation: %s", text_values.shape)
            logger.info("Label ids array shape before augmentation: %s", label_ids.shape)

            for l in backtrans_langs:
                text_values = np.append(text_values, df[f'preprocessed_{l}'].values, axis=0)
            label_ids = np.tile(label_ids, len(backtrans_langs) + 1)

            logger.info("Text values array shape AFTER augmentation: %s", text_values.shape)
            logger.info("Label ids array shape AFTER augmentation: %s", label_ids.shape)

        dataloader = tu.get_dataloader(self.config, self.tokenizer, text_values, label_ids, bs, **kwargs) 

        if include_df:
            return (dataloader, df)

        return dataloader        

    def __preprocess_df(self, df):
        df['label_id'] = df.category.map(self.LABEL_MAP)

        df['preprocessed'] = df.review.str.replace(r'\s+', ' ')
        all_backtrans_langs = list(set(self.config.train_backtrans_langs) | set(self.config.test_backtrans_langs))
        for l in all_backtrans_langs:
            df[f'preprocessed_{l}'] = df[f'review_{l}'].str.replace(r'\s+', ' ')

        print(df.preprocessed.values[:4])

        if all_backtrans_langs:
            if 'fr' in all_backtrans_langs:
                print(df.preprocessed_fr.values[:4])
            if 'de' in all_backtrans_langs:
                print(df.preprocessed_de.values[:4])

        return df

    def load_df(self):
        self.df = self.__preprocess_df(pd.read_csv(util.data_path("review_classification", "clap_backtrans.csv")))

    def get_train_valid_dataloaders(self, include_valid_df=False):
        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.config.seed)
        splits = kfold.split(self.df, y=self.df.label_id)

        for (train_idxs, valid_idxs) in splits:
            train_df = self.df.iloc[train_idxs]
            valid_df = self.df.iloc[valid_idxs]

            yield (self.get_dataloader(train_df, bs=self.config.train_bs, balance=True, backtrans_langs=self.config.train_backtrans_langs),
                   self.get_dataloader(valid_df, bs=self.config.eval_bs, shuffle=False, include_df=include_valid_df, backtrans_langs=self.config.test_backtrans_langs))