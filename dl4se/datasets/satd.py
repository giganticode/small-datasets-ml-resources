import dl4se.transformers.util as tu
from dl4se import util
import pandas as pd

from functools import cached_property

import numpy as np
import re

from dl4se.datasets.util import preprocess_comment, load_backtrans_dfs

class Dataset:

    RE_KEYWORDS = re.compile(r'(?:FIXME|TODO|TO DO|HACK|NOTE):?', re.IGNORECASE)
    BACKTRANS_LANGS = ['de', 'fr', 'ru', 'ja', 'it', 'ar', 'zh', 'ko']
    NEGATIVE_CLASS_NAME = 'WITHOUT_CLASSIFICATION'
    BINARY_LABEL_NAMES = ['NO_TECHNICAL_DEBT', 'TECHNICAL_DEBT']

    def __init__(self, config, binary=False, name=None, self_train_thresh=None, keyword_masking_frac=None):
        self.config = config
        self.binary = binary

        self.load_df()
        self.load_self_train_df(self_train_thresh)

    #     self.add_keyword_masked(keyword_masking_frac)

    # def add_keyword_masked(self, frac):
    #     if frac is not None:
    #         keyword_masked_df = self.df[(self.df.preprocessed.str.contains(Dataset.RE_KEYWORDS)) & (self.df.classification != 'WITHOUT_CLASSIFICATION')]
    #         print("masked shape:", keyword_masked_df.shape)
    #         print("frac:", frac)
    #         keyword_masked_df = keyword_masked_df.sample(frac=frac)
    #         print("masked shape:", keyword_masked_df.shape)
    #         print("masked before")
    #         print(keyword_masked_df.preprocessed.head())
    #         keyword_masked_df.preprocessed = keyword_masked_df.preprocessed.str.replace(Dataset.RE_KEYWORDS, '')
    #         print("masked after")
    #         print(keyword_masked_df.preprocessed.head())

    #         print("keyword masked")
    #         print(keyword_masked_df.head())

    #         keyword_masked_df.replace('', np.nan, inplace=True)
    #         keyword_masked_df.dropna(inplace=True)

    #         self.df = pd.concat([self.df, keyword_masked_df])

    def load_df(self):
        df = pd.read_csv(util.data_path(
            "satd", "dataset.csv"), delimiter=',')

        self.project_names = df.projectname.unique()
        self._label_names = sorted(df.classification.unique().tolist())
        # move WITHOUT_CLASSIFICATION to 0th position (so it has class_id 0)       
        self._label_names.remove(self.NEGATIVE_CLASS_NAME)
        self._label_names.insert(0, self.NEGATIVE_CLASS_NAME)

        self.label_map = self.build_label_map()

        if not self.binary:
            df = df[df.classification.isin(['DESIGN', 'IMPLEMENTATION', self.NEGATIVE_CLASS_NAME])]

        self.df = self.preprocess_df(df, self.label_map)

    def load_self_train_df(self, self_train_thresh):
        self.self_train_df = None

        if not self.binary:
            raise ValueError('self-training only available in binary mode')

        if self_train_thresh is not None:
            self.self_train_df = pd.read_csv(util.data_path("satd", "unclassified_pos.csv"), delimiter=',')
            print("self-train shape before filter", self.self_train_df.shape)
            self.self_train_df = self.self_train_df[self.self_train_df.uncertainty < self_train_thresh]
            self.self_train_df = self.preprocess_df(self.self_train_df, {'NO_TECHNICAL_DEBT': 0, 'TECHNICAL_DEBT': 1})

            print(self.self_train_df.head())
            self.self_train_df = self.self_train_df[['projectname', 'commenttext', 'preprocessed', 'label']]
            self.self_train_df.dropna(inplace=True)
            print(self.self_train_df.head())
            print("unclassified shape after filter", self.self_train_df.shape)

    def build_label_map(self):
        if self.binary:
            return {l: 0 if l == 'WITHOUT_CLASSIFICATION' else 1 for i, l in enumerate(self._label_names)}
        else:
            return {l: i for i, l in enumerate(self._label_names)}


    def backtrans(self, df):
        if not self.config.backtrans_langs:
            return df

        backtrans_dfs = load_backtrans_dfs('satd', self.config.backtrans_langs, 'dataset', index_col=0)

        positive_df = df[df.label != 0]
        index = positive_df.index

        def map_backtrans_df(backtrans_df):
            backtrans_df = self.preprocess_df(backtrans_df.loc[index], self.label_map)
            print('backtr shape', backtrans_df.shape)
            print(backtrans_df.label)
            print(positive_df.label)
            assert((backtrans_df.label == positive_df.label).all())
            assert((backtrans_df.projectname == positive_df.projectname).all())
            return backtrans_df

        backtrans_dfs = map(map_backtrans_df, backtrans_dfs)
        return pd.concat([df, *backtrans_dfs], ignore_index=True)


    def preprocess_df(self, df, label_map):
        df = df.assign(preprocessed = df.commenttext.map(preprocess_comment))
        df['label'] = df.classification.map(label_map)
        df.dropna(inplace=True)

        if self.config.remove_duplicates:
            df.drop_duplicates(('projectname', 'preprocessed'), inplace=True)

        return df

    def get_dataloader(self, tokenizer, df, bs, include_df=False, **kwargs):
        # convert_label = {'DEFECT': 0, 'DESIGN': 1,
        #                 'IMPLEMENTATION': 2, 'TEST': 3,
        #                 'WITHOUT_CLASSIFICATION': 4, 'DOCUMENTATION': 5}

        text_values = df.preprocessed.values
        label_ids = df.label.values

        print(label_ids.dtype)
        print(np.unique(label_ids))

        dataloader = tu.get_dataloader(self.config, tokenizer, text_values, label_ids, bs, **kwargs) 

        if include_df:
            return (dataloader, df)

        return dataloader

    @property
    def num_projects(self):
        return self.project_names.shape[0]

    @property
    def label_names(self):
        if self.binary:
            return Dataset.BINARY_LABEL_NAMES
        else:
            return self._label_names

    def get_fold_dataloaders(self, tokenizer, include_valid_df=False):
        for leave_out_project in self.project_names:
            yield (leave_out_project, *self.get_train_valid_dataloaders(tokenizer, leave_out_project, include_valid_df=include_valid_df))

    def get_complete_train_dataloader(self, tokenizer):
        train_df = self.df

        return self.get_dataloader(tokenizer, train_df,
                            bs=self.config.train_bs, balance=True)

    def build_train_df(self, leave_out_project):
        train_df = self.df[self.df.projectname != leave_out_project]

        if self.config.sample_frac:
            train_df = train_df.sample(frac=self.config.sample_frac)

        train_df = self.backtrans(train_df)

        if self.self_train_df is not None:
            train_df = train_df.append(self.self_train_df, ignore_index=True)

        return train_df

    def get_train_valid_dataloaders(self, tokenizer, leave_out_project, include_valid_df=False):
        if not leave_out_project in self.project_names:
            raise ValueError(f"invalid project {leave_out_project}")

        train_df = self.build_train_df(leave_out_project)
        valid_df = self.df[self.df.projectname == leave_out_project]

        # if self.backtrans_df is not None:
        #     train_df = pd.concat(
        #         [train_df, self.backtrans_df[self.backtrans_df.projectname != leave_out_project]], ignore_index=True)

        # if self.binary and self.self_train_df is not None:
        #     train_df = pd.concat([train_df, self.self_train_df], ignore_index=True)

        # train_df.to_csv('/tmp/test.csv', index=False)

        print(leave_out_project)
        print(train_df.shape)
        print(valid_df.shape)

        # assert(train_df.shape[0] + valid_df.shape[0] == self.df.shape[0])

        return (
            self.get_dataloader(tokenizer, train_df,
                                bs=self.config.train_bs, balance=True),
            self.get_dataloader(tokenizer, valid_df,
                                bs=self.config.eval_bs, include_df=include_valid_df))




