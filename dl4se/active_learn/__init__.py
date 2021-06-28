import torch
import torch.nn.functional as F
import math
import json

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

from dl4se.transformers.experiment import Experiment
import dl4se.transformers.util as tu
from dl4se import util
from dl4se.datasets.senti4sd import Dataset as SentiDataset, default_config

from dl4se.active_learn import acquisition_funcs as af

from dl4se.logging import logger

def active_learn(config, model_config, tokenizer, results, label_names, test_df, full_pool_df, backtrans_pool_dfs, get_dataloader_func, run_configs, active_learning_iters=10, dropout_iters=20, balance=False):
    test_dataloader = get_dataloader_func(test_df, bs=config.eval_bs)

    for run_config in run_configs:
        method, dropout, backtrans_langs, cluster_size = run_config
        run_name = method.__name__
        if dropout:
            run_name += '_dropout'
        run_name = '_'.join([run_name, *backtrans_langs, f"c{cluster_size}"])
        
        util.set_seed(config)

        model = tu.load_model(config, model_config)
        model.to(config.device)

        # remove initial seed from pool
        train_df, pool_df = train_test_split(
            full_pool_df, train_size=config.active_learn_seed_size, random_state=config.seed)

        logger.info("RUN CONFIG: %s (pool size: %d)", run_name, pool_df.shape[0])

        experiment = Experiment(config, model, tokenizer, label_names=label_names, run_name=run_name, results=results)

        cur_iter = 0


        extra_log = {'iter': cur_iter, 'pool': pool_df.shape[0]}
        experiment.evaluate('test', test_dataloader, extra_log=extra_log)

        while pool_df.shape[0] > 0:
            train_dataloader = get_dataloader_func(train_df, bs=config.train_bs, balance=balance)
            
            # DON'T SHUFFLE THE POOL!
            dataloader_pool = get_dataloader_func(pool_df, bs=config.eval_bs, shuffle=False)

            logger.info("=================== Remaining %d (%s) ================", pool_df.shape[0], run_config)
            logger.info("Evaluating: training set size: %d | pool set size: %d", train_df.shape[0], pool_df.shape[0])

            global_step, tr_loss = experiment.train(train_dataloader)

            extra_log = {'iter': cur_iter, 'pool': pool_df.shape[0]}

            _, _, preds = experiment.evaluate('pool', dataloader_pool, extra_log=extra_log)
            experiment.evaluate('test', test_dataloader, extra_log=extra_log)

            if method != af.random_conf:
                if dropout:
                    for i in range(dropout_iters):
                        torch.manual_seed(i)

                        _, _, preds_i = experiment.evaluate(
                            'pool_dropout', dataloader_pool, mc_dropout=True, skip_cb=True)
                        preds_i = torch.from_numpy(preds_i)
                        probs_i = F.softmax(preds_i, dim=1)

                        if i == 0:
                            probs = probs_i
                        else:
                            probs.add_(probs_i)
                    probs.div_(dropout_iters)
                else:
                    preds = torch.from_numpy(preds)
                    probs = F.softmax(preds, dim=1)
            else:
                preds = torch.from_numpy(preds)

                # only need the shape
                probs = preds

            scores = method(probs)
            _, topk_indices = torch.topk(scores, min(cluster_size * config.active_learn_step_size, scores.shape[0]))

            if cluster_size > 1:
                topk_preds = preds[topk_indices]
                n_clusters = min(config.active_learn_step_size, scores.shape[0])
                kmeans = KMeans(n_clusters=n_clusters).fit(topk_preds.numpy())
                _, unique_indices = np.unique(kmeans.labels_, return_index=True)
                topk_indices = topk_indices[unique_indices]
                # assert(topk_indices.shape[0] == n_clusters)
                logger.debug("top_k: %s", topk_indices.shape)

            logger.debug("%s %s", scores.shape, pool_df.shape)

            assert(scores.shape[0] == pool_df.shape[0])

            uncertain_rows = pool_df.iloc[topk_indices]
            train_df = train_df.append(uncertain_rows, ignore_index=True)

            for backtrans_lang in backtrans_langs:
                backtrans_pool_df = backtrans_pool_dfs[backtrans_lang]
                backtrans_uncertain_rows = backtrans_pool_df[backtrans_pool_df.id.isin(uncertain_rows.id)]
                train_df = train_df.append(backtrans_uncertain_rows, ignore_index=True)

            pool_df = pool_df.drop(pool_df.index[topk_indices])
            cur_iter += 1

        logger.debug("Pool exhausted, stopping active learning loop (%d remaining)", pool_df.shape[0])

        results = experiment.results
    return results