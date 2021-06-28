import random
import os
import io
import logging
import glob
import time
import json
import pprint

from collections import OrderedDict

from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler, WeightedRandomSampler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn

import torch.nn.functional as F

import pandas as pd

#from pytorch_lamb import Lamb

from transformers import AdamW, WEIGHTS_NAME, get_linear_schedule_with_warmup, get_constant_schedule
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, BertPreTrainedModel, BertModel
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
#from transformers import AlbertTokenizer, AlbertConfig, AlbertForSequenceClassification

from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score, average_precision_score

from dl4se import util
from dl4se.logging import logger, console
# import util
# from util import Config
#import wandb

# from transformer_util import *

#os.environ['WANDB_API_KEY'] = '2f019c7c0fdede8bf5f26fd4553e6b22cc43de94'
# wandb.init(project="senti4sd-bert")


class Experiment(object):
    def __init__(self, config, model, tokenizer, total_samples=None, label_names=None, results=None, run_name=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        self.global_step = 0

        self.optimizer_state_dict = None
        self.scheduler_state_dict = None

        self.total_samples = total_samples

        self.label_names = label_names

        self.results = results
        self.run_name = run_name

        util.set_seed(config)

    __prfs_names = ['precision', 'recall', 'f1', 'support']
    __report_metrics = ['acc', 'macro_f1', 'micro_f1', 'macro_auc', 'avg_precision']

    def after_eval_cb(self, eval_name, result, pred_label_ids, preds, extra_log):

        row = OrderedDict(step=self.global_step,
                          eval_name=eval_name,
                          run_name=self.run_name)

        row.update(extra_log)

        for key in self.__report_metrics:
            if key in result:
               row[key] = result[key]

        prfs = result['prfs']

        for metric_idx, metric_name in enumerate(self.__prfs_names):
            for label_idx, label_name in enumerate(self.label_names):
                col_name = f"{label_name}_{metric_name}"
                row[col_name] = result['prfs'][metric_idx][label_idx]

        if self.config.seeds:
            row['seed'] = self.config.seed

        if self.results is None:
            logger.warning("Creating new results DataFrame")
            self.results = pd.DataFrame(row, columns=row.keys(), index=[0])
        else:
            logger.debug("Adding row: %s", row)
            self.results = self.results.append(row, ignore_index=True)

        if self.config.get('out_file', None):
            self.results.to_csv(self.config.out_file, index=False)
            # results = self.results
            # key = self.run_name
            # if key not in results:
            #     results[key] = {}
            # if eval_name not in results[key]:
            #     results[key][eval_name] = {}
            # results[key][eval_name][self.global_step] = result
            # with open(self.config.out_file, 'w') as f:
            #     json.dump(results, f, indent=4, cls=util.ExtendedJSONEncoder)

    def after_logging(self, result):
        pass

    def train(self, train_dataloader, valid_dataloader=None, test_dataloader=None, should_continue=False):
        """ Train the model """
        tb_writer = SummaryWriter()

        train_epochs = self.config.train_epochs

        if self.config.max_steps > 0:
            train_steps = self.config.max_steps
            train_epochs = self.config.max_steps // (
                len(train_dataloader) // self.config.grad_acc_steps) + 1
        else:
            train_steps = len(
                train_dataloader) // self.config.grad_acc_steps * train_epochs

        if self.total_samples and should_continue:
            steps_total = self.total_samples // self.config.train_bs // self.config.grad_acc_steps * train_epochs
        else:
            steps_total = train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=self.config.lr, eps=self.config.adam_eps, )


        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=steps_total
        )

        # self.scheduler = get_constant_schedule(self.optimizer)

        if should_continue and self.global_step > 0:
            logger.info("loading saved optimizer and scheduler states")
            assert(self.optimizer_state_dict)
            assert(self.scheduler_state_dict)
            self.optimizer.load_state_dict(self.optimizer_state_dict)
            self.scheduler.load_state_dict(self.scheduler_state_dict)
        else:
            logger.info("Using fresh optimizer and scheduler")

        if self.config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.config.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.config.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d (%d)", len(
            train_dataloader.dataset), len(train_dataloader))
        logger.info("  Num Epochs = %d", train_epochs)
        logger.info("  Batch size = %d",
                         self.config.train_bs)
        logger.info("  Learning rate = %e", self.config.lr)
        logger.info("  Loss label weights = %s", self.config.loss_label_weights)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.config.train_bs
            * self.config.grad_acc_steps
        )
        logger.info("  Gradient Accumulation steps = %d",
                         self.config.grad_acc_steps)
        logger.info("  Total optimization steps = %d", train_steps)

        if not should_continue:
            self.global_step = 0

        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # # Check if continuing training from a checkpoint
        # if os.path.exists(self.config.model_path):
        #     if self.config.should_continue:
        #         step_str = self.config.model_path.split("-")[-1].split("/")[0]

        #         if step_str:
        #             # set self.global_step to gobal_step of last saved checkpoint from model path
        #             self.global_step = int(step_str)
        #             epochs_trained = self.global_step // (len(train_dataloader) //
        #                                                   self.config.grad_acc_steps)
        #             steps_trained_in_current_epoch = self.global_step % (
        #                 len(train_dataloader) // self.config.grad_acc_steps)

        #             logger.info(
        #                 "  Continuing training from checkpoint, will skip to saved self.global_step")
        #             logger.info(
        #                 "  Continuing training from epoch %d", epochs_trained)
        #             logger.info(
        #                 "  Continuing training from global step %d", self.global_step)
        #             logger.info("  Will skip the first %d steps in the first epoch",
        #                         steps_trained_in_current_epoch)

        train_loss = 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(train_epochs), desc="Epoch",
        )
        util.set_seed(self.config)  # Added here for reproductibility

        self.model.train()

        if self.config.train_head_only:
            for param in self.model.roberta.embeddings.parameters():
                param.requires_grad = False
            logger.info("Training only head")
            # for param in self.model.__getattr__(self.config.model_type).roberta.parameters():
            #     param.requires_grad = False

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()

                inputs = self.__inputs_from_batch(batch)
                outputs = self.model(**inputs)

                # model outputs are always tuple in transformers (see doc)
                loss = outputs[0]

                if self.config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.config.grad_acc_steps > 1:
                    loss = loss / self.config.grad_acc_steps

                if self.config.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                batch_loss = loss.item()
                train_loss += batch_loss
                
                if (step + 1) % self.config.grad_acc_steps == 0:
                    if self.config.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer), self.config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    self.global_step += 1

                    if self.config.logging_steps > 0 and self.global_step % self.config.logging_steps == 0:
                        logs = {}
                        if valid_dataloader:
                            result_valid, * \
                                _ = self.evaluate(
                                    'valid', valid_dataloader, backtrans=(test_dataloader == None))
                            logs.update({f"valid_{k}": v for k, v in result_valid.items()})

                        if test_dataloader:
                            test_dataloader = test_dataloader if isinstance(test_dataloader, dict) else {'test': test_dataloader}
                            for eval_name, dataloader_or_tuple in test_dataloader.items():
                                if isinstance(dataloader_or_tuple, tuple):
                                    dataloader, kwargs = dataloader_or_tuple
                                else:
                                    dataloader = dataloader_or_tuple
                                    kwargs = {}

                                result_test, * \
                                    _ = self.evaluate(
                                        eval_name, dataloader, **kwargs)
                                logs.update({f"{eval_name}_{k}": v for k, v in result_test.items()})

                        learning_rate_scalar = self.scheduler.get_last_lr()[0]
                        logger.info("Learning rate: %f (at step %d)", learning_rate_scalar, step)
                        logs["learning_rate"] = learning_rate_scalar
                        logs["train_loss"] = train_loss

                        self.after_logging(logs)

                        logger.info("Batch loss: %f", batch_loss)

                        # for key, value in logs.items():
                        #     tb_writer.add_scalar(key, value, self.global_step)

                    if self.config.save_steps > 0 and self.global_step % self.config.save_steps == 0:
                        # Save model checkpoint
                        self.save_checkpoint()

                if self.config.max_steps > 0 and self.global_step > self.config.max_steps:
                    epoch_iterator.close()
                    break
            if self.config.max_steps > 0 and self.global_step > self.config.max_steps:
                train_iterator.close()
                break

        if self.config.train_head_only:
            logger.info("Training only head")
            # for param in self.model.__getattr__(self.config.model_type).parameters():
            #     param.requires_grad = True

            for param in self.model.roberta.embeddings.parameters():
                param.requires_grad = False

        tb_writer.close()
        self.optimizer_state_dict = self.optimizer.state_dict()
        self.scheduler_state_dict = self.scheduler.state_dict()

        avg_train_loss = train_loss / self.global_step

        logger.info("Learning rate now: %s", self.scheduler.get_last_lr())
        logger.info("***** Done training *****")
        return self.global_step, avg_train_loss

    def save_model(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        logger.info("Saving model to %s", model_path)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.config.as_dict(), os.path.join(model_path, "training_config.bin"))

    def save_checkpoint(self):
        output_dir = os.path.join(
            self.config.output_model_path, "checkpoint-{}".format(self.global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            self.model.module if hasattr(
                self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(self.config.as_dict(), os.path.join(
            output_dir, "training_self.config.bin"))
        logger.info(
            "Saving model checkpoint to %s", output_dir)

        torch.save(self.optimizer.state_dict(), os.path.join(
            output_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(
            output_dir, "scheduler.pt"))
        logger.info(
            "Saving optimizer and scheduler states to %s", output_dir)

    def predict(self, dataloader):
        self.model.eval()

        preds = None

        for batch in tqdm(dataloader, desc="Predicting"):
            batch = tuple(t.to(self.config.device) for t in batch)

            input_ids, attention_mask, _ = batch

            with torch.no_grad():
                inputs = {
                    "input_ids": input_ids,
                     "attention_mask": attention_mask
                }
                # if config.model_type != "distilbert":
                #    inputs["token_type_ids"] = (
                #        batch[2] if config.model_type in [
                #            "bert", "xlnet", "albert"] else None
                #    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = self.model(**inputs)
                logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        return preds

    def logits_to_label_ids(self, logits):
        if not self.config.multi_label:
            label_ids = np.argmax(logits, axis=1)
        else:
            label_ids = F.sigmoid(torch.from_numpy(logits)).numpy() > 0.5
        return label_ids            

    def evaluate(self, eval_name, dataloader, mc_dropout=False, skip_cb=False, pred_label_ids_func=None, backtrans=True, extra_log={}):
        dropout_ps = {}

        def set_dropout_to_train(m):
            if type(m) == nn.Dropout:
                logger.info("setting dropout into train mode (%s)", str(m))
                logger.info(
                    "setting dropout into train mode (%s)", str(m))
                m.p = 0.5
                m.train()

        def reset_dropout_to_eval(m):
            if type(m) == nn.Dropout:
                p = dropout_ps[m]
                logger.info(
                    "reseting dropout into eval mode (%s) p=%d", str(m), p)
                m.p = p
                m.eval()

        # Eval!
        logger.info("***** Running evaluation %s*****", eval_name)
        logger.info("  Num examples = %d", len(dataloader.dataset))
        logger.info("  Batch size = %d", self.config.eval_bs)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        true_label_ids = None

        self.model.eval()

        if mc_dropout:
            self.model.apply(set_dropout_to_train)

        for batch in tqdm(dataloader, desc="Evaluating"):

            with torch.no_grad():
                inputs = self.__inputs_from_batch(batch)
                labels = inputs['labels']

                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                true_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                true_label_ids = np.append(
                    true_label_ids, labels.detach().cpu().numpy(), axis=0)

        if mc_dropout:
            self.model.apply(reset_dropout_to_eval)

        eval_loss = eval_loss / nb_eval_steps

        if self.config.test_backtrans_langs and backtrans:
            logger.info('Using test augmentation...')
            groups = np.split(preds, len(self.config.test_backtrans_langs)+1)
            #preds = sum(groups)

            preds = np.mean(groups, axis=0)
            #preds = np.maximum.reduce(groups)
            true_label_ids = true_label_ids[:preds.shape[0]]

        label_idxs = list(range(len(self.label_names)))

        if self.config.soft_label:
            true_label_ids = np.argmax(true_label_ids, axis=1)

        pred_label_ids = self.logits_to_label_ids(preds)

        if pred_label_ids_func:
            pred_label_ids = pred_label_ids_func(pred_label_ids)

        # print(out_label_ids)
        # print(max_preds)
        # print(out_label_ids.shape, max_preds.shape)

        result = {'acc': accuracy_score(true_label_ids, pred_label_ids),
                  'macro_f1': f1_score(true_label_ids, pred_label_ids, average='macro'),
                  'micro_f1': f1_score(true_label_ids, pred_label_ids, average='micro'),
                  'prfs': precision_recall_fscore_support(true_label_ids, pred_label_ids, labels=label_idxs)}                          

        if not self.config.multi_label:
            result['cm'] = confusion_matrix(true_label_ids, pred_label_ids).ravel()

        if self.config.num_labels == 2:
            result['macro_auc'] = roc_auc_score(true_label_ids, pred_label_ids, average='macro')
            result['avg_precision'] = average_precision_score(true_label_ids, pred_label_ids)

        logger.info("***** Eval results {} *****".format(eval_name))

        try:
            logger.info("\n %s", classification_report(
                true_label_ids, pred_label_ids,
                labels=label_idxs,
                target_names=self.label_names,
            ))

            result['report'] = classification_report(
                true_label_ids, pred_label_ids,
                labels=label_idxs,
                target_names=self.label_names,
                output_dict=True)
        except ValueError as e:
            print(e)
            pass

        logger.info("\n Accuracy = %f", result['acc'])

        if self.config.num_labels == 2:
            logger.info("\n MacroAUC = %f", result['macro_auc'])
            logger.info("\n AUPRC = %f", result['avg_precision'])

        logger.info("***** Done evaluation *****")

        if not skip_cb:
            self.after_eval_cb(eval_name, result, pred_label_ids, preds, extra_log)
        return result, pred_label_ids, preds

    def __inputs_from_batch(self, batch, labels=True):
        batch = tuple(t.to(self.config.device) for t in batch)
        input_ids, attention_mask, label_ids, *rest = batch

        if rest:
            extra_features = rest[0]
        else:
            extra_features = None

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "extra_features": extra_features}
        if labels:
            inputs["labels"] = label_ids

        # if self.config.model_type != "distilbert":
        #    inputs["token_type_ids"] = (
        #        batch[2] if self.config.model_type in [
        #            "bert", "xlnet", "albert"] else None
        #    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

        #         outputs = model(b_input_ids, token_type_ids=None,
        #                         attention_mask=b_input_mask, labels=b_labels)

        return inputs

    def interpret(self, dataloader, df, label_names=None):

        dataset = dataloader.dataset
        sampler = SequentialSampler(dataset)

        # We need a sequential dataloader with bs=1
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=4)

        logger.info("***** Running interpretation *****")
        logger.info("  Num examples = %d", len(dataset))

        # preds = None
        losses = None
        pred_labels = []

        self.model.eval()

        for batch in tqdm(dataloader, desc="Interpretation"):

            with torch.no_grad():
                inputs = self.__inputs_from_batch(batch)
                # if config.model_type != "distilbert":
                #    inputs["token_type_ids"] = (
                #        batch[2] if config.model_type in [
                #            "bert", "xlnet", "albert"] else None
                #    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = self.model(**inputs)
                batch_loss, logits = outputs[:2]

                if self.config.n_gpu > 1:
                    batch_loss = batch_loss.mean()  # mean() to average on multi-gpu parallel training

                batch_loss = batch_loss.detach().cpu().view(1)

                pred_label_ids = self.logits_to_label_ids(logits.detach().cpu())
                pred_label_id = pred_label_ids[0]
                if label_names:
                    pred_labels.append(label_names[pred_label_id])
                else:
                    pred_labels.append(pred_label_id)

            if losses is None:
                # preds = logits.detach().cpu().numpy()
                losses = batch_loss
            else:
                # preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                losses = torch.cat((losses, batch_loss), dim=0)


        top_values, top_indices = torch.topk(losses, 100)
        top_indices = top_indices.numpy()
        top_pred_labels = [pred_labels[top_index] for top_index in top_indices]

        top_df = df.iloc[top_indices]
        top_df = top_df.assign(loss=top_values.numpy(), pred_label=top_pred_labels)

        return top_df
# def train_eval(config):
#     if not config.do_train and not config.do_eval:
#         return

#     if config.local_rank not in [-1, 0]:
#         # Make sure only the first process in distributed training will download model & vocab
#         torch.distributed.barrier()

#     tokenizer = load_tokenizer(config)
#     model = load_model(config)

#     if config.local_rank == 0:
#         # Make sure only the first process in distributed training will download model & vocab
#         torch.distributed.barrier()

#     model.to(config.device)

#     logger.info("Training/evaluation parameters %s", config)

#     train_dataloader, valid_dataloader = get_train_valid_dataloaders(
#         config, tokenizer)
#     test_dataloader = get_test_dataloader(config, tokenizer)

#     # Train
#     if config.do_train:
#         self.global_step, tr_loss = train(
#             config, train_dataloader, model, tokenizer, valid_dataloader, test_dataloader)
#         logger.info(" self.global_step = %s, average loss = %s",
#                     self.global_step, tr_loss)

#         if config.save_model:
#             # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
#             if (config.local_rank == -1 or torch.distributed.get_rank() == 0):
#                 # Create output directory if needed
#                 if not os.path.exists(config.output_model_path) and config.local_rank in [-1, 0]:
#                     os.makedirs(config.output_model_path)

#                 logger.info("Saving model checkpoint to %s",
#                             config.output_model_path)
#                 # Save a trained model, configuration and tokenizer using `save_pretrained()`.
#                 # They can then be reloaded using `from_pretrained()`
#                 model_to_save = (
#                     model.module if hasattr(model, "module") else model
#                 )  # Take care of distributed/parallel training
#                 model_to_save.save_pretrained(config.output_model_path)
#                 tokenizer.save_pretrained(config.output_model_path)

#                 # Good practice: save your training arguments together with the trained model
#                 torch.save(config.as_dict(), os.path.join(
#                     config.output_model_path, "training_config.bin"))

#                 # Load a trained model and vocabulary that you have fine-tuned
#                 #model = model_cls.from_pretrained(config.output_model_path)
#                 #tokenizer = tokenizer_cls.from_pretrained(config.output_model_path, do_lower_case=config.do_lower_case)
#                 # model.to(config.device)

#     results = {}

#     if config.do_eval:
#         # Evaluation

#         if config.local_rank in [-1, 0]:
#             if not config.do_train:
#                 tokenizer = load_tokenizer(config)

#             model_cls = get_model_cls(config)

#             if config.do_train and config.eval_all_checkpoints:
#                 checkpoints = list(
#                     os.path.dirname(c) for c in sorted(glob.glob(config.output_model_path + "/**/" + WEIGHTS_NAME, recursive=True))
#                 )
#                 logging.getLogger("transformers.modeling_utils").setLevel(
#                     logging.WARN)  # Reduce logging
#                 logger.info(
#                     "Evaluate the following checkpoints: %s", checkpoints)
#                 for checkpoint in checkpoints:
#                     self.global_step = checkpoint.split(
#                         "-")[-1] if len(checkpoints) > 1 else ""
#                     prefix = checkpoint.split(
#                         "/")[-1] if checkpoint.find("checkpoint") != -1 else ""

#                     model = model_cls.from_pretrained(checkpoint)
#                     model.to(config.device)
#                     result, *_ = self.evaluate("test_checkpoint", test_dataloader, model=model, tokenizer=tokenizer)
#                     result = dict((k + "test_{}".format(self.global_step), v)
#                                   for k, v in result.items())
#                     results.update(result)
#             else:
#                 model_name = config.output_model_path if config.do_train else config.model_path
#                 if not config.do_train:
#                     model = model_cls.from_pretrained(model_name)
#                     model.to(config.device)

#                 result, *_ = self.evaluate("final_test", test_dataloader, model=model, tokenizer=tokenzier)
#                 results.update(result)


# def simple_run(config):
#     config.logging_eval_test = True
#     config.logging_steps = 90
#     config.train_size = 0.7
#     config.valid_size = 0.3

#     tokenizer = load_tokenizer(config)
#     model = load_model(config)
#     model.to(config.device)

#     train_dataloader, valid_dataloader = get_train_valid_dataloaders(
#         config, tokenizer)
#     test_dataloader = get_test_dataloader(config, tokenizer)

#     train(
#         config, train_dataloader, model, tokenizer,
#         valid_dataloader, test_dataloader
#     )

#     result, *_ = evaluate(config, model, tokenizer,
#                           test_dataloader, "test")


# def main(config):
#     init_all(config)
#     # train_eval(config)
#     # simple_run(config)
#     # valid_test_corr.main(config)
#     active_learn.main(config)

    # experiment(config)
