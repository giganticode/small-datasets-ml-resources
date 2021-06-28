
import html
from bs4 import BeautifulSoup
import re
from dl4se import util

import pandas as pd
from sklearn.model_selection import train_test_split

RE_SYMBOLS = re.compile(r'[\<\>;@~#^!&=/\\\'\"\-\+\*\[\]\{\}\(\)\|]')
RE_REPEATED_SPACE = re.compile(r'\s+')
RE_REPEATED_STAR = re.compile(r'\*+')

def train_valid_split(*dfs, config):
    return train_test_split(*dfs, train_size=config.train_size, test_size=config.valid_size, random_state=config.seed)

def load_backtrans_dfs(dir_name, langs, prefix, **kwargs):
    print(kwargs)
    return {lang: pd.read_csv(util.data_path(dir_name, f"{prefix}_{lang}.csv"), **kwargs) for lang in langs}

def preprocess_comment_soft(comment):
    comment = comment.replace('/*', '')
    comment = comment.replace('*/', '')
    comment = comment.replace('* ', ' ')
    comment = comment.replace(' *', ' ')
    comment = comment.replace('*\n', ' ')
    comment = comment.replace('\n*', ' ')
    comment = RE_REPEATED_STAR.sub(' ', comment)
    comment = RE_REPEATED_SPACE.sub(' ', comment)
    comment = comment.strip()
    return comment


def preprocess_comment(comment, strip_html=True):
    if strip_html and '<' in comment:
        soup = BeautifulSoup(comment, 'lxml')
        comment = soup.get_text()

    comment = html.unescape(comment)
    comment = comment.replace('\t', ' ')
    comment = comment.replace('\n', ' ')
    comment = comment.replace('...', ' ')
    comment = comment.replace('.html', '')
    comment = RE_SYMBOLS.sub(' ', comment)
    comment = RE_REPEATED_SPACE.sub(' ', comment)
    comment = comment.strip()

    if comment.isspace() or comment == '':
        comment = None

    return comment
