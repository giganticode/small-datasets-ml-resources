
import dl4se.config as config
import dl4se.util as util


def set_defaults(config):
    config.train_epochs = 4
    config.lr = 4e-5
    config.model_path = 'huggingface/CodeBERTa-small-v1'
    config.train_bs = 16
    config.eval_bs = 128
    config.max_seq_len = 512
    config.num_labels = 10
    config.hidden_dropout_prob = 0.01
    config.multi_label = False
    config.loss_label_weights = None
    config.logging_steps = 10

def parse_args(parser):
    parser.add_argument("--sep_token", default=False, action='store_true', required=False)
    # parser.add_argument("--balance", default=False, action='store_true', required=False)
    # parser.add_argument("--comment_last", default=False, action='store_true', required=False)

def after_parse_func(config):
    pass

def get_config(experiment_parse_args_func=None):
    return config.get_config(set_defaults, parse_args, experiment_parse_args_func=experiment_parse_args_func, after_parse_func=after_parse_func)
