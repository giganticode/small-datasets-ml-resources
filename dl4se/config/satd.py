
import dl4se.config as config
import dl4se.util as util


def set_defaults(config):
    config.train_epochs = 8
    config.lr = 4e-5
    config.max_seq_len = 255
    config.train_bs = 32
    config.eval_bs = 128 * 2
    config.num_labels = 2
    config.hidden_dropout_prob = 0.1
    config.logging_steps = 200
    # config.max_steps = 10
    # config.lr = 1e-4
    config.model_path = util.models_path('StackOBERTflow-comments-small-v1')

    # config.single_project = None
    # config.self_train_thresh = None
    # config.interp_output = None

def parse_args(parser):
    parser.add_argument("--self_train_thresh", default=None, type=float, required=False)
    parser.add_argument("--keyword_masking_frac", default=None, type=float, required=False)
    parser.add_argument("--remove_duplicates", default=False, action='store_true', required=False)
    parser.add_argument("--backtrans_langs", default=[], nargs='+', type=str, required=False)
    parser.add_argument("--sample_frac", default=None, type=float, required=False)

def get_config(experiment_parse_args_func=None):
    return config.get_config(set_defaults, parse_args, experiment_parse_args_func=experiment_parse_args_func)
