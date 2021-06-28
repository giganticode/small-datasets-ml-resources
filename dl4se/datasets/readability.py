import dl4se.transformers.util as tu
from dl4se import util
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from dl4se.logging import logger


from itertools import permutations

class Dataset:
    LABEL_NAMES2 = ['non_readable', 'readable']
    LABEL_NAMES3 = ['low', 'medium', 'high']

    LABEL_THRESHS = {
        'scalabrino': 3.6,
        'buse': 3.27, # 3.14
        'dorn': 3.4,
    }

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.all_bins = set()

        self.load_dfs()

    @property
    def n_splits(self):
        return 10

    @property
    def label_names(self):
        if self.config.num_labels == 3:
            return self.LABEL_NAMES3
        elif self.config.num_labels == 2:
            return self.LABEL_NAMES2
        else:            
            raise ValueError()

    def get_dataloader(self, df, features_df, bs, include_df=False, **kwargs):
        text_values = df.preprocessed.values
        label_ids = df.label.values

        if features_df is not None:
            print(features_df.columns)
            extra_features = features_df.drop('Readable', axis=1).to_numpy()
            extra_features = StandardScaler().fit_transform(extra_features)
            assert(extra_features.shape[1] == self.config.extra_features_size)
        else:
            extra_features = None

        # dataloader = tu.get_dataloader(self.config, self.tokenizer, text_values, label_ids, bs, extra_features=extra_features, **kwargs) 

        if self.config.sep_token:
            # leave text section empty
            dataloader = tu.get_dataloader(self.config, self.tokenizer, [''] * len(text_values), label_ids, bs, text_pair_values=text_values, **kwargs) 
        else:            
            dataloader = tu.get_dataloader(self.config, self.tokenizer, text_values, label_ids, bs, **kwargs) 

        if include_df:
            return (dataloader, df)

        return dataloader


    def __preprocess_df(self, df, name):

        if self.config.num_labels == 3 and name == 'scalabrino':
            label = df.avg.map(lambda x: 0 if x < 3 else 1 if x < 4 else 2)
        else:
            label = df.avg.map(lambda x: 1 if x > 3.6 else 0)

        df = df.assign(preprocessed=df.snippet.map(self.__preprocess), label=label)

        return df

    def __preprocess(self, snippet):

        if not self.config.line_length_tokens:
            return snippet

        lines = snippet.splitlines()
        line_lens = [len(line.rstrip()) for line in lines]
        bins = np.digitize(line_lens, np.linspace(60, 120, 3))

        for i, line in enumerate(lines):
            lines[i] = line + f'<l{bins[i]}>'

        self.all_bins.update(bins)

        return '\n'.join(lines)

    def __preprocess_features_df(self, df):
        return df.fillna(0)

    def load_dfs(self):
        self.scalabrino_df = self.__preprocess_df(pd.read_csv(util.data_path("readability", "scalabrino.csv")), 'scalabrino')
        #self.buse_df = self.__preprocess_df(pd.read_csv(util.data_path("readability", "buse.csv")), 'buse')
        #self.dorn_df = self.__preprocess_df(pd.read_csv(util.data_path("readability", "dorn.csv")), 'dorn')


        if self.config.features:
            self.scalabrino_features_df = self.__preprocess_features_df(pd.read_csv(util.data_path("readability", "features", "scalabrino_reduced.csv")))
            self.buse_features_df = self.__preprocess_features_df(pd.read_csv(util.data_path("readability", "features", "buse.csv")))
            self.dorn_features_df = self.__preprocess_features_df(pd.read_csv(util.data_path("readability", "features", "dorn.csv")))

            assert(self.scalabrino_df.shape[0] == self.scalabrino_features_df.shape[0])
            assert(self.buse_df.shape[0] == self.buse_features_df.shape[0])
            assert(self.dorn_df.shape[0] == self.dorn_features_df.shape[0])

            assert(self.scalabrino_df.label.equals(self.scalabrino_features_df.Readable))
            assert(self.buse_df.label.equals(self.buse_features_df.Readable))
        else:
            self.scalabrino_features_df = None
            self.buse_features_df = None
            self.dorn_features_df = None

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(pd.DataFrame({'label': self.scalabrino_df.label.eq(self.scalabrino_features_df.Readable), 'avg_rank': self.scalabrino_df.avg_rank}))
        #     print(pd.DataFrame({'label': self.buse_df.label.eq(self.buse_features_df.Readable), 'avg_rank': self.buse_df.avg_rank}))

        # assert(self.dorn_df.label.equals(self.dorn_features_df.Readable))

        # python = self.dorn_df[self.dorn_df.lang == 'python']
        # java = self.dorn_df[self.dorn_df.lang == 'java']
        # cuda = self.dorn_df[self.dorn_df.lang == 'cuda']

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     for p in permutations([cuda, python, java]):
        #         self.dorn_df = pd.concat(p, ignore_index=True)
        #         print("\n\n")
        #         eq = self.dorn_df.label.eq(self.dorn_features_df.Readable)
        #         print(eq.value_counts())
        #         print(pd.DataFrame({'eq': eq, 'avg_rank': self.dorn_df.avg_rank}))


        if self.config.line_length_tokens:
            self.__add_special_tokens()

    def __add_special_tokens(self):
        bins = [f'<l{b}>'for b in self.all_bins]
        logger.info(f'Adding special tokens {bins}')
        self.tokenizer.add_special_tokens({'additional_special_tokens': bins})

    def _get_train_valid_dataloaders(self, df, features_df, include_valid_df):
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.config.seed)
        splits = kfold.split(df)

        for (train_idxs, valid_idxs) in splits:
            train_df = df.iloc[train_idxs]
            valid_df = df.iloc[valid_idxs]

            if self.config.features:
                features_train_df = features_df.iloc[train_idxs]
                features_valid_df = features_df.iloc[valid_idxs]
            else:
                features_train_df = None       
                features_valid_df = None


            yield (self.get_dataloader(train_df, features_train_df, bs=self.config.train_bs, balance=False),
                   self.get_dataloader(valid_df, features_valid_df, bs=self.config.eval_bs, include_df=include_valid_df))

    def get_scalabrino_train_valid_dataloaders(self, include_valid_df=False):
        return self._get_train_valid_dataloaders(self.scalabrino_df, self.scalabrino_features_df, include_valid_df)

    def get_buse_train_valid_dataloaders(self, include_valid_df=False):
        return self._get_train_valid_dataloaders(self.buse_df, self.buse_features_df, include_valid_df)

    def get_dorn_train_valid_dataloaders(self, include_valid_df=False):
        return self._get_train_valid_dataloaders(self.dorn_df, self.dorn_features_df, include_valid_df)

    def get_combined_train_valid_dataloaders(self, include_valid_df=False):
        combined = pd.concat([self.scalabrino_df, self.buse_df, self.dorn_df], ignore_index=True)
        return self._get_train_valid_dataloaders(combined, include_valid_df)

