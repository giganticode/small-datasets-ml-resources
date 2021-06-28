
import dl4se.config as config
import dl4se.util as util


def set_defaults(config):
    config.train_epochs = 30
    config.lr = 3e-5
    config.model_type = None
    config.model_path = 'huggingface/CodeBERTa-small-v1'
    config.train_bs = 32
    config.eval_bs = 256
    config.max_seq_len = 512
    config.num_labels = 2
    config.hidden_dropout_prob = 0.1
    config.multi_label = False
    config.logging_steps = 5
    config.do_lower_case = False

def parse_args(parser):
    parser.add_argument("--features", default=False, action="store_true")
    parser.add_argument("--line_length_tokens", default=False, action="store_true")
    parser.add_argument("--num_labels", type=int, choices=[2,3], default=2)
    parser.add_argument("--sep_token", default=False, action='store_true', required=False)


def after_parse_func(config):
    if config.features:
        config.extra_features_size = 13

def get_config(experiment_parse_args_func=None):
    return config.get_config(set_defaults, parse_args, experiment_parse_args_func=experiment_parse_args_func, after_parse_func=after_parse_func)
