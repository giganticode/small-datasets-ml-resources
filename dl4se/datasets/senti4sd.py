import dl4se.transformers.util as tu
from dl4se import util
import pandas as pd

import dl4se.datasets.util as du

class Dataset(object):

    # DO NOT CHANGE ORDER!
    LABEL_MAP = {'negative': 0, 'positive': 1, 'neutral': 2}
    LABEL_MAP_SIGN = {-1: 0, 1: 1, 0: 2}
    LABEL_NAMES = ['negative', 'positive', 'neutral']

    ALL_BACKTRANS_LANGS = ['de', 'fr', 'it', 'ja', 'ko', 'ru', 'zh', 'ar', 'es', 'hu', 'tr']

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.load_dfs()

    @property
    def label_names(self):
        return self.LABEL_NAMES

    def get_dataloader(self, df, bs, text_col='text', **kwargs):
        text_values = df[text_col].values

        if self.config.soft_label:
            label_ids = df[self.LABEL_NAMES].to_numpy()
        else:
            label_ids = df.label.values

        dataloader = tu.get_dataloader(self.config, self.tokenizer, text_values, label_ids, bs, **kwargs) 
        return dataloader

    def load_dfs(self):
        train_valid_df = pd.read_csv(util.data_path("senti4sd", "train_ext.csv"))
        test_df = pd.read_csv(util.data_path("senti4sd", "test_ext.csv"))

        self.train_valid_df = self.preprocess(train_valid_df)
        self.test_df = self.preprocess(test_df)

    def load_backtrans_dfs(self, langs, prefix):
        backtrans_dfs = du.load_backtrans_dfs("senti4sd", langs, prefix, sep=';')
        backtrans_dfs = [self.preprocess(df) for lang, df in backtrans_dfs.items()]
        return backtrans_dfs

    def get_train_valid_dataloaders(self):
        train_valid_backtrans_dfs = self.load_backtrans_dfs(
            self.config.train_backtrans_langs, 'train')

        train_valid_dfs = du.train_valid_split(
            self.train_valid_df, *train_valid_backtrans_dfs, config=self.config)

        train_df = pd.concat(train_valid_dfs[::2], ignore_index=True)
        valid_df = pd.concat(train_valid_dfs[1::2], ignore_index=True)

        assert len(set(train_df.id) & set(valid_df.id)) == 0

        return (self.get_dataloader(train_df, bs=self.config.train_bs),
                self.get_dataloader(valid_df, bs=self.config.eval_bs, shuffle=False))

    def get_active_learn_pool_df(self, backtrans_langs):
        train_valid_backtrans_dfs = self.load_backtrans_dfs(backtrans_langs, 'train')
        return pd.concat([self.train_valid_df, *train_valid_backtrans_dfs])

    def neutral_to_negative(self, pred_label_ids):
        pred_label_ids[pred_label_ids == self.LABEL_MAP['neutral']] = self.LABEL_MAP['negative']
        return pred_label_ids

    def get_test_dataloader(self):
        # test_df_orig = pd.read_csv(util.data_path("sentidata/app_reviews.csv"), delimiter=';')
        test_backtrans_dfs = self.load_backtrans_dfs(
            self.config.test_backtrans_langs, 'test')
        test_df = pd.concat([self.test_df, *test_backtrans_dfs], ignore_index=True)

        return self.get_dataloader(test_df, bs=self.config.eval_bs, shuffle=False)

    def get_stack_overflow_dataloader(self):
        test_df = pd.read_csv(util.data_path("sentidata/StackOverflow.csv"), delimiter=',')
        test_df = self.preprocess(test_df, 'oracle', label_map=self.LABEL_MAP_SIGN)

        return self.get_dataloader(test_df, bs=self.config.eval_bs, shuffle=False)

    def get_jira_dataloader(self):
        test_df = pd.read_csv(util.data_path("sentidata/jira.csv"), delimiter=',')
        #NOTE: Replaces the label column with the preprocessed one
        test_df = self.preprocess(test_df, 'label', label_map=self.LABEL_MAP_SIGN)

        return self.get_dataloader(test_df, bs=self.config.eval_bs, shuffle=False)

    def get_app_reviews_dataloader(self):
        test_df = pd.read_csv(util.data_path("sentidata/AppReviews.csv"), delimiter=',')
        test_df = self.preprocess(test_df, 'oracle', label_map=self.LABEL_MAP_SIGN)

        return self.get_dataloader(test_df, bs=self.config.eval_bs, text_col='sentence', shuffle=False)

    def preprocess(self, df, col='polarity', label_map=LABEL_MAP):
        if self.config.soft_label:
            for label_id, label in enumerate(self.LABEL_NAMES):
                if self.config.smooth_hard_label or (not label in df.columns):
                    df[label] = (df[col].map(label_map) == label_id).astype(float)

                if self.config.smooth_label:
                    df[label] = df[label] * (1 - self.config.smooth_label) + self.config.smooth_label / self.config.num_labels

        else:
            df['label'] = df[col].map(label_map)

        print(df.head(20))

        return df

def default_config():
    config = tu.default_config()

    util.init_config(config)

    return config