from transformers import PreTrainedModel, PretrainedConfig
from transformers import BertConfig, BertPreTrainedModel, BertModel, BertTokenizer
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import AutoConfig

from torch.nn import CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss, BCEWithLogitsLoss
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict

from dl4se.logging import logger

class EnumeratedDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return (index, *self.dataset[index])

    def __len__(self):
        return len(self.dataset)

class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        logprobs = F.log_softmax(input, dim=1)
        loss = -torch.sum(target * logprobs, dim=1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')


class FocalLoss(nn.Module):
    
    def __init__(self, gamma=3., **nll_args):
        super().__init__()
        self.gamma = gamma
        self.nll_args = nll_args
        
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)

        prob_ = 1 - prob
        prob_.pow_(self.gamma)
        prob_.mul_(log_prob)

        return F.nll_loss(
            # ((1 - prob) ** self.gamma) * log_prob, 
            prob_,
            target, 
            **self.nll_args)

class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        dense_size = config.hidden_size + config.extra_features_size
        self.dense = nn.Linear(dense_size, dense_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(dense_size, config.num_labels)

    def forward(self, features, extra_features=None, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])

        if extra_features is not None:
            x = torch.cat((x, extra_features), dim=1)

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class HeadlessConfig(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.head_cls = kwargs.pop('head_cls', None)
        self.label_weights = kwargs.pop('label_weights', None)
        self.multi_label = kwargs.pop('multi_label', False)
        self.soft_label = kwargs.pop('soft_label', False)
        self.loss_func = kwargs.pop('loss_func', 'cross_entropy')
        self.extra_features_size = kwargs.pop('extra_features_size', 0)
        self.device = kwargs.pop('device', 'cuda')

class HeadlessBertConfig(HeadlessConfig, BertConfig):
    def __init__(self, **kwargs):
        # super(HeadlessBertConfig, self).__init__(**kwargs)
        HeadlessConfig.__init__(self, **kwargs)
        BertConfig.__init__(self, **kwargs)

class HeadlessRobertaConfig(HeadlessConfig, RobertaConfig):
    def __init__(self, **kwargs):
        # super(HeadlessRobertaConfig, self).__init__(**kwargs)
        HeadlessConfig.__init__(self, **kwargs)
        RobertaConfig.__init__(self, **kwargs)
        # print(kwargs)
        # raise RuntimeError(str(vars(self)))


class ModelUtilsMixin():
    def reinit_layers(self, layer_idxs):
        for layer_idx in layer_idxs:
            logger.debug(f'reinitializing layer {layer_idx}')
            layer = self.get_layer(layer_idx)
            layer.apply(self._init_weights)

    def reinit_pooler(self):
        self.get_pooler().apply(self._init_weights)

class HeadlessModelForSequenceClassification(ModelUtilsMixin):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.loss_func, *loss_func_params = config.loss_func.split(':')

        try:
            self.loss_func_params = [float(param) for param in loss_func_params]
        except ValueError as exp:
            raise ValueError(f"invalid loss function parameters") from exp

        logger.info('using loss function: %s', self.loss_func)

        if config.label_weights:
            self.label_weights = torch.tensor(config.label_weights)
            logger.info('using label weights: %s', self.label_weights)
        else:
            self.label_weights = None            

        self.multi_label = config.multi_label
        self.soft_label = config.soft_label
        self.classifier = None
        self.classifier = eval(config.head_cls)(config)
        self.head = eval(config.head_cls)(config)

    def calculate_loss(self, logits, labels):
        if self.label_weights is not None:
            self.label_weights = self.label_weights.to(logits.device)

        if self.multi_label:
            if self.loss_func == 'cross_entropy' or self.loss_func == 'bce':
                loss_func = BCEWithLogitsLoss(weight=self.label_weights)
            elif self.loss_func == 'soft_margin':
                loss_func = MultiLabelSoftMarginLoss(weight=self.label_weights)
            else:
                self.__raise_invalid_loss_func()

            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        elif self.soft_label:
            loss_func = SoftLabelCrossEntropyLoss()
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        elif self.num_labels == 1:
            #  We are doing regression
            loss_func = MSELoss()
            loss = loss_func(logits.view(-1), labels.view(-1))
        else:
            if self.loss_func == 'cross_entropy':
                loss_func = CrossEntropyLoss(weight=self.label_weights)
            elif self.loss_func == 'focal':
                loss_func_params = dict(zip(['gamma'], self.loss_func_params))
                loss_func = FocalLoss(weight=self.label_weights, **loss_func_params)
            else:
                self.__raise_invalid_loss_func()

            loss = loss_func(
                logits.view(-1, self.num_labels), labels.view(-1))    
        return loss                

    def __raise_invalid_loss_func(self):
        raise ValueError(f"invalid loss function '{self.loss_func}'")

class HeadlessRobertaForSequenceClassification(HeadlessModelForSequenceClassification, BertPreTrainedModel):
    config_class = HeadlessRobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        BertPreTrainedModel.__init__(self, config)
        HeadlessModelForSequenceClassification.__init__(self, config)
        # super(HeadlessRobertaForSequenceClassification, self).__init__(config)

        self.roberta = RobertaModel(config)

        self.init_weights()

    def get_layer(self, idx):
        return self.roberta.base_model.encoder.layer[idx]

    def get_pooler(self):
        return self.roberta.base_model.pooler

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        extra_features=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.head(sequence_output, extra_features=extra_features)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss = self.calculate_loss(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



class BertClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, extra_features=None, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HeadlessBertForSequenceClassification(HeadlessModelForSequenceClassification, BertPreTrainedModel):
    def __init__(self, config):
        # super(HeadlessBertForSequenceClassification, self).__init__(config)
        BertPreTrainedModel.__init__(self, config)
        HeadlessModelForSequenceClassification.__init__(self, config)

        self.bert = BertModel(config)

        self.init_weights()

    def get_layer(self, idx):
        return self.bert.base_model.encoder.layer[idx]

    def get_pooler(self):
        return self.bert.base_model.pooler

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, extra_features=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]
        logits = self.head(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss = self.calculate_loss(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class AutoHeadlessConfig:

    MODEL_CLASSES = {
        "bert": (HeadlessBertConfig, HeadlessBertForSequenceClassification, BertTokenizer, BertClassificationHead),
        "roberta": (HeadlessRobertaConfig, HeadlessRobertaForSequenceClassification, RobertaTokenizer, RobertaClassificationHead)
    }

    @classmethod
    def model_class(cls, model_config):
        return AutoHeadlessConfig.MODEL_CLASSES[model_config.model_type][1]

    @classmethod
    def tokenizer_class(cls, model_config):
        return AutoHeadlessConfig.MODEL_CLASSES[model_config.model_type][2]

    @classmethod
    def head_class(cls, model_config):
        return AutoHeadlessConfig.MODEL_CLASSES[model_config.model_type][3]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict:
            model_type = config_dict["model_type"]
            config_class = AutoHeadlessConfig.MODEL_CLASSES[model_type][0]

            if not "head_cls" in kwargs:
                kwargs['head_cls'] = AutoHeadlessConfig.MODEL_CLASSES[model_type][3].__name__

            return config_class.from_dict(config_dict, **kwargs)
        else:
            raise ValueError(f"Unrecognized model {pretrained_model_name_or_path} (missing 'model_type').")