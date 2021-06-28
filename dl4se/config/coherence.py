
import dl4se.config as config
import dl4se.util as util


def set_defaults(config):
    config.train_epochs = 20
    config.lr = 3e-5
    config.model_path = 'huggingface/CodeBERTa-small-v1'
    config.train_bs = 32
    config.eval_bs = 256
    config.max_seq_len = 512
    config.num_labels = 2
    config.hidden_dropout_prob = 0.1
    config.multi_label = False
    config.loss_label_weights = None
    config.logging_steps = 20
    config.do_lower_case = False

def parse_args(parser):
    parser.add_argument("--balance", default=True, action='store_true', required=False)
    parser.add_argument("--comment_last", default=False, action='store_true', required=False)
    parser.add_argument("--sep_token", default=False, action='store_true', required=False)

def after_parse_func(config):
    pass

def get_config(experiment_parse_args_func=None):
    return config.get_config(set_defaults, parse_args, experiment_parse_args_func=experiment_parse_args_func, after_parse_func=after_parse_func)
