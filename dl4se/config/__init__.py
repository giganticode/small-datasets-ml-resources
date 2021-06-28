import time
import argparse
import torch
from dl4se import util

class Config(dict):
    def __init__(self, dict={}):
        dict.__init__(self)
        self._writable = False

    def __enter__(self):
        self._writable = True
        return self

    def __exit__(self, type, value, tb):
        self._writable = False

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, val):
        if attr == '_writable':
            object.__setattr__(self, attr, val)
        else:
            if not self._writable and not attr in self:
                raise AttributeError(attr)
            self[attr] = val

    def as_dict(self):
        return self

    def copy(self, **kwargs):
        copy = Config(self)
        copy.update(kwargs)
        return copy

def init_config(config, parse_args_func=None, experiment_parse_args_func=None, after_parse_func=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=None, type=float, required=False)
    parser.add_argument("--eval_bs", default=None, type=int, required=False)
    parser.add_argument("--train_bs", default=None, type=int, required=False)
    parser.add_argument("--max_steps", default=None, type=int, required=False)
    parser.add_argument("--train_epochs", default=None, type=int, required=False)
    parser.add_argument("--logging_steps", default=None, type=int, required=False)
    parser.add_argument('--loss_label_weights', nargs='+', type=float, required=False)
    parser.add_argument('--loss_func', default='cross_entropy', type=str, required=False)
    parser.add_argument('--seed', type=int, default=42, required=False)
    parser.add_argument('--seeds', nargs='+', type=int, required=False)
    parser.add_argument('--reinit_layers', nargs='+', type=int, default=[], required=False)
    parser.add_argument("--reinit_pooler", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--n_gpu", default=1, type=int)
    parser.add_argument("--out_file", default=None, type=str, required=False)

    parser.add_argument("--no_pretrain", action="store_true")

    default_output_model_path = util.models_path("{}-{}".format(
        list(filter(None, config.model_path.split("/"))).pop(),
        time.strftime("%Y_%m_%d-%H_%M_%S")
    ))

    parser.add_argument("--output_model_path",
                        default=default_output_model_path,
                        type=str,
                        required=False)

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name"
    )

    parser.add_argument(
        "--tokenizer_path",
        default=None,
        type=str,
        required=False
    )

    if parse_args_func:
        parse_args_func(parser)

    if experiment_parse_args_func:
      experiment_parse_args_func(parser)

    args = parser.parse_args()
    for var, value in vars(args).items():
        if value or not var in config:
            config[var] = value

    config['device'] = "cuda" if not config.no_cuda else "cpu"

    if after_parse_func:
        after_parse_func(config)    

def default_config():
    config = Config()

    with config:
        config.train_backtrans_langs = []
        config.test_backtrans_langs = []
        config.train_size = None
        config.warmup_steps = 0
        config.adam_eps = 1e-8
        config.seed = 1234
        config.seeds = None
        config.model_type = 'bert'
        config.fp16 = False
        config.fp16_opt_level = 'O1'
        config.local_rank = -1
        config.grad_acc_steps = 1
        config.save_steps = 0  # 10_000
        config.eval_all_checkpoints = False
        config.should_continue = False
        config.max_steps = -1
        config.weight_decay = 0.0
        config.no_cuda = False
        config.max_grad_norm = 1.0
        config.logging_steps = 100
        config.logging_eval_test = False
        config.logging_eval_valid = True
        config.save_model = False
        config.do_train = False
        config.do_eval = False
        config.do_lower_case = True
        config.tokenizer_path = None
        config.overwrite_output_model = False
        config.active_learn_seed_size = 0.05
        config.train_head_only = False
        config.multi_label = False
        config.soft_label = False
        config.loss_label_weights = None
        config.no_pretrain = False
        config.extra_features_size = 0
        config.n_gpu = 1

    # util.init_config(config)

    return config


def get_config(set_defaults_func, parse_args_func, after_parse_func=None, *args, **kwargs):
  config = default_config()
  with config:
    set_defaults_func(config)

  init_config(config, *args, parse_args_func=parse_args_func, after_parse_func=after_parse_func, **kwargs)
  return config
