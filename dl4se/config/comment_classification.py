
import dl4se.config as config
import dl4se.util as util


def set_defaults(config):
    config.train_epochs = 8
    config.lr = 4e-5
    config.max_seq_len = 512
    config.train_bs = 32
    config.eval_bs = 128 * 2
    config.num_labels = 16
    config.hidden_dropout_prob = 0.1
    config.logging_steps = 200
    config.model_path = 'giganticode/StackOBERTflow-comments-small-v1'

def parse_args(parser):
    parser.add_argument("--balance", default=False, action='store_true', required=False)

def get_config(experiment_parse_args_func=None):
    return config.get_config(set_defaults, parse_args, experiment_parse_args_func=experiment_parse_args_func)
