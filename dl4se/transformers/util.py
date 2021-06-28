import os
import io
import time
import json
import logging

from tqdm import tqdm, trange
import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW, WEIGHTS_NAME, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, BertPreTrainedModel, BertModel
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
# from transformers import AlbertTokenizer, AlbertConfig, AlbertForSequenceClassification

from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, classification_report
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from dl4se.transformers.ext import HeadlessBertForSequenceClassification, HeadlessBertConfig, BertClassificationHead, RobertaClassificationHead
from dl4se.transformers.ext import HeadlessRobertaForSequenceClassification, HeadlessRobertaConfig, RobertaClassificationHead
from dl4se.transformers.ext import AutoHeadlessConfig, EnumeratedDataset

from dl4se import util
from dl4se.logging import logger

import inspect

from scipy.stats.mstats import gmean

MODEL_CLASSES = {
    "bert": (HeadlessBertConfig, HeadlessBertForSequenceClassification, BertTokenizer, BertClassificationHead),
    # "bert": (BertConfig, (BertForSequenceClassification, WeightedBertForSequenceClassification), BertTokenizer),
    # "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (HeadlessRobertaConfig, HeadlessRobertaForSequenceClassification, RobertaTokenizer, RobertaClassificationHead),
    # "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    # "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    # "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    # "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}


def build_attention_masks(id_sents):
    # [PAD] has id 0
    return [[float(token_id > 0) for token_id in id_sent] for id_sent in id_sents]

def get_dataloader(config, tokenizer, text_values, label_ids, bs, text_pair_values=None, shuffle=True, balance=False, enumerated=False, extra_features=None):

    # for t in text_values:
    #     print(t)
    #     x = tokenizer.tokenize(t)
    #     print(x)

    # for t, l in zip(text_values, label_ids):
    #     x = tokenizer.tokenize(t)[255:]
    #     if x: print(l, x)

    # logger.info('Original: %s', sents_list[0])
    # logger.info('Tokenized: %s', tokenizer.tokenize(sents_list[0]))
    # logger.info('Token IDs: %s', tokenizer.convert_tokens_to_ids(
    #     tokenizer.tokenize(sents_list[0])))

    input_ids = [tokenizer.encode(t,
                                  text_pair=text_pair_values[i] if text_pair_values is not None else None,
                                  add_special_tokens=True,
                                  max_length=config.max_seq_len,
                                  truncation=True,
                                  pad_to_max_length=True) for i, t in enumerate(text_values)]

    logger.debug(tokenizer.decode(input_ids[0]))

    attention_masks = build_attention_masks(input_ids)

    input_ids_t = torch.tensor(input_ids)
    label_ids_t = torch.tensor(label_ids, dtype=(
        torch.float32 if config.multi_label or config.soft_label else torch.int64))
    attention_masks_t = torch.tensor(attention_masks)

    tensors = [input_ids_t, attention_masks_t, label_ids_t]

    if extra_features is not None:
        extra_features_t = torch.tensor(extra_features, dtype=torch.float32)
        tensors.append(extra_features_t)

    dataset = TensorDataset(*tensors)

    if enumerated:
        dataset = EnumeratedDataset(dataset)

    if config.local_rank != -1:
        sampler = DistributedSampler(dataset)
    elif balance:
        if config.soft_label:
            raise ValueError('balancing for soft labels in not implemented')
        if not config.multi_label:
            label_weights = 1.0 / np.bincount(label_ids)
        else:
            # label values are boolean
            label_weights = 1.0 / np.sum(label_ids, axis=0)
        logger.info('label weights %s', label_weights / label_weights.sum())
        if not config.multi_label:
            weights = [label_weights[l] for l in label_ids]
        else:
            weights = (label_ids_t * torch.tensor(label_weights)).mean(dim=1)
        sampler = WeightedRandomSampler(weights, int(1*len(weights)))
    elif shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    logger.info(f"Using sampler: {sampler}")
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=bs, num_workers=4)

    return dataloader


def load_tokenizer(config, model_config):
    # tokenizer_cls = MODEL_CLASSES[config.model_type][2]
    tokenizer_cls = AutoHeadlessConfig.tokenizer_class(model_config)
    tokenizer = tokenizer_cls.from_pretrained(
        config.tokenizer_path if config.tokenizer_path else config.model_path,
        #do_lower_case=config.do_lower_case,
        cache_dir=None,
    )
    return tokenizer

def load_model_config(config):
    model_config = AutoHeadlessConfig.from_pretrained(config.model_path,
        num_labels=config.num_labels,
        hidden_dropout_prob=config.hidden_dropout_prob,
        output_attentions=False,
        output_hidden_states=False,
        cache_dir=None,
        label_weights=config.loss_label_weights,
        loss_func=config.loss_func,
        multi_label=config.multi_label,
        soft_label=config.soft_label,
        extra_features_size=config.extra_features_size,
        device=config.device
    )
    return model_config

def load_model(config, model_config):
    # config_cls, _, _, head_class = MODEL_CLASSES[config.model_type]

    # model_cls = get_model_cls(config)

    model_cls = AutoHeadlessConfig.model_class(model_config)
    head_class = AutoHeadlessConfig.head_class(model_config)

    logger.debug("loaded model_config: %s", model_config)

    if config.no_pretrain:
        logger.warning("Using non pretrained model!")
        model = model_cls(config=model_config)
    else:
        model = model_cls.from_pretrained(
            config.model_path,
            config=model_config
        )

        if config.reinit_layers:
            logger.info(f"reinitializing layers... {config.reinit_layers}")
            model.reinit_layers(config.reinit_layers)

        if config.reinit_pooler:
            logger.info(f"reinitializing pooler...")
            model.reinit_pooler()

    return model




