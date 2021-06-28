import dl4se.transformers.util as tu
from dl4se import util
import pandas as pd
import numpy as np

from dl4se.logging import logger
from sklearn.model_selection import train_test_split

from itertools import permutations

class Dataset:
    LABEL_NAMES = [str(i) for i in range(10)]

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        df = pd.read_csv(util.data_path("complexity", "cmpx.csv"))
        df = self.__preprocess_df(df)

        print(df.head())

        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
        train_df, valid_df = train_test_split(train_df, test_size=0.2, shuffle=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

    def __preprocess_df(self, df):
        # # let the tokenizer join comment and code with a separator token
        # df['preprocessed'] = '/* ' + df.comment + ' */\n' + df.code
        # print(df.head(2))
        return df

    @property
    def label_names(self):
        return self.LABEL_NAMES

    def get_dataloader(self, df, bs, include_df=False, fake=False, **kwargs):
        text_values = df.methodText.values
        label_ids = df.cyclomatic.values
        if fake:
            label_ids = label_ids.copy()[:128]
            text_values = text_values[:128]
            np.random.shuffle(label_ids)

        if self.config.sep_token:
            # leave text section empty
            dataloader = tu.get_dataloader(self.config, self.tokenizer, [''] * len(text_values), label_ids, bs, text_pair_values=text_values, **kwargs) 
        else:            
            dataloader = tu.get_dataloader(self.config, self.tokenizer, text_values, label_ids, bs, **kwargs) 

        if include_df:
            return (dataloader, df)
        return dataloader        

    def get_train_dataloader(self):
        return self.get_dataloader(self.train_df, bs=self.config.train_bs)

    def get_valid_dataloader(self, include_df=False):
        return self.get_dataloader(self.valid_df, bs=self.config.eval_bs, include_df=include_df)

    def get_fake_valid_dataloader(self, include_df=False):
        return self.get_dataloader(self.valid_df, bs=self.config.eval_bs, include_df=include_df, fake=True)

    def get_test_dataloader(self):
        return self.get_dataloader(self.test_df, bs=self.config.eval_bs)