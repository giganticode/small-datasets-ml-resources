import torch
import util
import json

from transformer_util import *

def main(config):
    data = {}

    config.logging_eval_test = True
    config.logging_steps = 25

    def logging_cb(eval_name, result, max_preds, preds):
        results = data[p][n].get(eval_name)
        if results is None:
          results = []
          data[p][n][eval_name] = results
        results.append(result)

        with open(config.out_file, 'w') as f:
            json.dump(data, f, indent=4, cls=util.ExtendedJSONEncoder)

    # [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None]:
    for p in [0.5, 0.7, 0.8]:
        data[p] = {}

        # for n in range(2):
        for n in range(4):
            config.train_size = p
            config.valid_size = 1.0 - p

            config.seed = n * 100
            util.set_seed(config)

            data[p][n] = {'valid': None, 'train': None}

            #config.valid_size = 1 - p if p else 0.05
            #config.seed = n

            # util.set_seed(config)

            tokenizer = load_tokenizer(config)
            model = load_model(config)
            model.to(config.device)

            train_dataloader, valid_dataloader = get_train_valid_dataloaders(
                config, tokenizer)
            test_dataloader = get_test_dataloader(config, tokenizer)

            train(
                config, train_dataloader, model, tokenizer,
                valid_dataloader, test_dataloader, logging_cb=logging_cb
            )
            # result, *_ = evaluate(config, model, tokenizer,
            #                       test_dataloader, logging_cb)

            # data[p][n] = result


    # wandb.log(results)
