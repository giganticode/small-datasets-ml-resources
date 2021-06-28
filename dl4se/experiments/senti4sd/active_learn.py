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
from dl4se.config.senti4sd import get_config
from dl4se.active_learn import active_learn, acquisition_funcs as af

RUN_CONFIGS = {
    "all_af_c3": [
        (af.least_conf, False, [], 3),
        (af.margin_conf, False, [], 3),
        (af.ratio_conf, False, [], 3),
        (af.entropy, False, [], 3),
        (af.random_conf, False, [], 3),
    ],

    "all_af_c1": [
        (af.least_conf, False, [], 1),
        (af.margin_conf, False, [], 1),
        (af.ratio_conf, False, [], 1),
        (af.entropy, False, [], 1),
        (af.random_conf, False, [], 1),
    ],

    "all_af_fr_de_it_c3": [
        (af.least_conf, False, ['fr', 'de', 'it'], 3),
        (af.margin_conf, False, ['fr', 'de', 'it'], 3),
        (af.ratio_conf, False, ['fr', 'de', 'it'], 3),
        (af.random_conf, False, ['fr', 'de', 'it'], 3),
        (af.entropy, False, ['fr', 'de', 'it'], 3),
    ],

    "random_all_langs": [
        (af.random_conf, False, [], 1),
        (af.random_conf, False, ['fr'], 1),
        (af.random_conf, False, ['zh'], 1),
        (af.random_conf, False, ['fr', 'de'], 1),
        (af.random_conf, False, ['zh', 'ja'], 1),
        (af.random_conf, False, ['fr', 'de', 'it'], 1),
        (af.random_conf, False, ['zh', 'ja', 'ko'], 1),
        (af.random_conf, False, ['fr', 'de', 'it', 'es', 'zh', 'ja', 'ko', 'ru', 'ar', 'hu', 'tr'], 1),
    ]
}

def parse_args(parser):
    parser.add_argument("--run_configs", choices=RUN_CONFIGS.keys(), type=str, required=True)

def main(config, results):
    model_config = tu.load_model_config(config)
    tokenizer = tu.load_tokenizer(config, model_config)

    ds = SentiDataset(config, tokenizer)
    backtrans_dfs = ds.load_backtrans_dfs(ds.ALL_BACKTRANS_LANGS, 'train')
    run_configs = RUN_CONFIGS[config.run_configs]
    results = active_learn(config, model_config, tokenizer, results, ds.label_names, ds.test_df, ds.train_valid_df, backtrans_dfs, ds.get_dataloader, run_configs)
    return results

    # active_learn(config, )
    # test_dataloader = ds.get_test_dataloader()

    # model = tu.load_model(config, model_config)
    # model.to(config.device)
    # util.set_seed(config)

    # train_dataloader, valid_dataloader = ds.get_train_valid_dataloaders()
    # test_dataloader = ds.get_test_dataloader()

    # test_dataloaders = {'test': ds.get_test_dataloader()}

    # if config.jira:
    #     test_dataloaders['JIRA'] = (ds.get_jira_dataloader(), dict(pred_label_ids_func=ds.neutral_to_negative))

    # if config.app_reviews:
    #     test_dataloaders['AppReviews'] = ds.get_app_reviews_dataloader()

    # if config.sentidata_so:
    #     test_dataloaders['StackOverflow (SentiData)'] = ds.get_stack_overflow_dataloader()

    # experiment = Experiment(config, model, tokenizer, label_names=ds.label_names, results=results)
    # global_step, tr_loss = experiment.train(train_dataloader, valid_dataloader=valid_dataloader, test_dataloader=test_dataloaders)

    # interp_df = experiment.interpret(test_dataloader, ds.test_df, label_names=ds.label_names)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(interp_df)

    # if config.interp_out_file:
    #     interp_df.to_csv(config.interp_out_file, index=False)

    # return experiment.results

config = get_config(parse_args)
util.run_with_seeds(config, main)



# class ActiveLearn(Experiment):
#     def after_eval_cb(self, eval_name, result, max_preds, preds):
#         d = self.d
#         results = d.results
#         key = d.method.__name__
#         if d.mc_dropout:
#             key += '_dropout'
#         key += '_'.join(d.backtrans_langs)
#         if key not in results:
#             results[key] = {'test': {}, 'pool': {}}
#         results[key][eval_name][d.p] = result
#         with open(self.config.out_file, 'w') as f:
#             json.dump(results, f, indent=4, cls=util.ExtendedJSONEncoder)


# def main():

#     k = 240
#     dropout_iters = 20
#     active_learning_iters = 10

#     config = default_config()
#     config.logging_steps = -1
#     config.active_learn_seed_size = k
#     config.lr = 5e-5
#     config.seed = 54321
#     # config.train_epochs = 3
#     # config.train_head_only = True

#     pd.set_option('display.max_rows', None)

#     # config.model_type = 'roberta'
#     # config.model_path = util.models_path('StackOBERTflow-comments-small-v1')

#     #config.model_path = util.models_path('stackoverflow_1M_large')
#     # config.train_head_only = True
#     tu.parse_args(config)


#     tokenizer = tu.load_tokenizer(config)

#     ds = SentiDataset(config, tokenizer)
#     test_dataloader = ds.get_test_dataloader()

#     train_valid_df = ds.train_valid_df

#     # for method in [least_conf, margin_conf, ratio_conf, entropy, random_conf]:
#     #for run_config in [(least_conf, False, []), (random_conf, False, [])]:
#     #for run_config in [(least_conf, False, [])]:
#     #for run_config in [(least_conf, False, ['uda1', 'uda2', 'uda3']), (least_conf, False, ['fr', 'de', 'es']), (least_conf, False, []), (random_conf, False, [])]:
#     for run_config in [(least_conf, False, ['uda1', 'uda2', 'uda3'], 1),
#                        (least_conf, False, ['uda1', 'uda2', 'uda3'], 3),
#                        (least_conf, False, ['fr', 'de', 'es'], 1),
#                        (least_conf, False, ['fr', 'de', 'es'], 3),
#                        (least_conf, False, [], 1),
#                        (least_conf, False, [], 3),
#                        (random_conf, False, [], 1)]:
#     # for run_config in [(least_conf, False, [], 8), (least_conf, False, [], 4), (least_conf, False, [], 2), (least_conf, False, [], 1), (random_conf, False, [], 4), (random_conf, False, [], 1)]:
#     #for run_config in [(least_conf, False, ['uda1']), (least_conf, False, ['fr']), (least_conf, False, []), (random_conf, False, [])]:
#         method, dropout, backtrans_langs, cluster_size = run_config
#         run_name = method.__name__
#         if dropout:
#             run_name += '_dropout'
#         run_name = '_'.join([run_name, *backtrans_langs, f"c{cluster_size}"])
        
#         util.set_seed(config)

#         model = tu.load_model(config)
#         model.to(config.device)

#         # with d:
#         #     d.mc_dropout = dropout
#         #     d.method = method
#         #     d.backtrans_langs = backtrans_langs


#         train_valid_backtrans_dfs = ds.load_backtrans_dfs(backtrans_langs, 'train')

#         train_pool_df = pd.concat([train_valid_df, *train_valid_backtrans_dfs])

#         train_df, pool_df = train_test_split(
#             train_pool_df, train_size=config.active_learn_seed_size)

#         experiment = Experiment(config, model, tokenizer, label_names=ds.label_names,
#                                 run_name=run_name)
#         # experiment = ActiveLearn(
#         #     config, model, tokenizer, total_samples=(train_pool_df.shape[0] * (1 + len(backtrans_langs))))

#         # while pool_df.shape[0] > k:
#         for _ in range(active_learning_iters):

#             if pool_df.shape[0] < k:
#                 break

#             train_dataloader = ds.get_dataloader(train_df, bs=config.train_bs)
            
#             # DON'T SHUFFLE THE POOL
#             dataloader_pool = ds.get_dataloader(pool_df, bs=config.eval_bs, shuffle=False)

#             experiment.logger.info("=================== Remaining %d (%s) ================", pool_df.shape[0], run_config)
#             experiment.logger.info("Evaluating: training set size: %d | pool set size: %d",
#                                    train_df.shape[0], pool_df.shape[0])

#             global_step, tr_loss = experiment.train(train_dataloader)

#             _, _, preds = experiment.evaluate('pool', dataloader_pool)
#             # write_result('valid', result_valid)

#             experiment.evaluate('test', test_dataloader)
#             # write_result('test', result_test)

#             if method != random_conf:
#                 if dropout:
#                     for i in range(dropout_iters):
#                         torch.manual_seed(i)

#                         _, _, preds_i = experiment.evaluate(
#                             'pool_dropout', dataloader_pool, mc_dropout=True, skip_cb=True)
#                         preds_i = torch.from_numpy(preds_i)
#                         probs_i = F.softmax(preds_i, dim=1)

#                         if i == 0:
#                             probs = probs_i
#                         else:
#                             probs.add_(probs_i)
#                     probs.div_(dropout_iters)
#                 else:
#                     preds = torch.from_numpy(preds)
#                     probs = F.softmax(preds, dim=1)
#             else:
#                 preds = torch.from_numpy(preds)

#                 # only need the shape
#                 probs = preds

#             scores = method(probs)
#             _, topk_indices = torch.topk(scores, min(cluster_size * k, scores.shape[0]))
#             topk_preds = preds[topk_indices]

#             if cluster_size > 1:
#                 n_clusters = min(k, scores.shape[0])
#                 kmeans = KMeans(n_clusters=n_clusters).fit(topk_preds.numpy())
#                 _, unique_indices = np.unique(kmeans.labels_, return_index=True)
#                 topk_indices = topk_indices[unique_indices]
#                 # assert(topk_indices.shape[0] == n_clusters)
#                 print('top_k', topk_indices.shape)


#             print(scores.shape, pool_df.shape)

#             assert(scores.shape[0] == pool_df.shape[0])

#             uncertain_rows = pool_df.iloc[topk_indices]
#             # uncertain_rows_backtrans = [
#             #     backtrans_df[backtrans_df.id.isin(uncertain_rows.id)]
#             #     for backtrans_df in train_valid_backtrans_dfs
#             # ]
#             # uncertain_rows = pd.concat([uncertain_rows, *uncertain_rows_backtrans])

#             print("\n\n")

#             #train_df = train_df.append(uncertain_rows, ignore_index=True)
#             train_df = train_df.append(uncertain_rows, ignore_index=True)
#             pool_df = pool_df.drop(pool_df.index[topk_indices])


# main()