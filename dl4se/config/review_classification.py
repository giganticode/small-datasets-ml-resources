
import dl4se.config as config
import dl4se.util as util


def set_defaults(config):
    config.train_epochs = 8
    config.lr = 5e-5
    config.max_seq_len = 256
    config.train_bs = 48
    config.eval_bs = 128 * 4
    config.num_labels = 4
    config.hidden_dropout_prob = 0.1
    config.logging_steps = 20
    config.model_path = 'giganticode/StackOBERTflow-comments-small-v1'
    config.active_learn_step_size = 120

def parse_args(parser):
    parser.add_argument("--train_backtrans_langs", default=[], nargs='+', type=str, required=False)
    parser.add_argument("--test_backtrans_langs", default=[], nargs='+', type=str, required=False)
    parser.add_argument("--clap", default=False, action='store_true', required=False)

def after_parse_func(config):
    if config.clap:
        config.num_labels = 7

def get_config(experiment_parse_args_func=None):
    return config.get_config(set_defaults, parse_args, after_parse_func=after_parse_func, experiment_parse_args_func=experiment_parse_args_func)
