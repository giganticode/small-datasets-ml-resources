
import dl4se.config as config
import dl4se.util as util

def set_defaults(config):
    config.train_epochs = 4
    config.lr = 5e-5
    # config.model_path = util.models_path('android.stackexchange.com')
    # config.model_path = util.models_path('android.stackexchange.com')#a
    # config.model_path = util.models_path('stackoverflow_1M')
    # config.model_path = util.models_path('StackOBERTflow-comments-small-v1/')
    # config.model_path = 'bert-large-uncased'
    #config.model_path = util.models_path('StackOBERTflow-comments-small-v1')
    config.model_path = 'giganticode/StackOBERTflow-comments-small-v1'
    config.max_seq_len = 255
    config.train_bs = 12
    config.eval_bs = 128 * 2
    config.hidden_dropout_prob = 0.07
    config.num_labels = 3
    config.logging_steps = 20
    config.active_learn_step_size = 180
    #config.max_steps = 1
    # config.max_steps = 10
    # config.lr = 1e-4

    config.valid_size = 0.3
    config.train_size = None

def parse_args(parser):
    parser.add_argument("--train_backtrans_langs", default=[], nargs='+', type=str, required=False)
    parser.add_argument("--test_backtrans_langs", default=[], nargs='+', type=str, required=False)
    parser.add_argument("--train_size", default=None, type=float, required=False)
    parser.add_argument("--soft_label", action="store_true", default=False)
    parser.add_argument("--smooth_hard_label", action="store_true", default=False)
    parser.add_argument("--smooth_label", default=None, type=float)

def get_config(experiment_parse_args_func=None):
    return config.get_config(set_defaults, parse_args, experiment_parse_args_func=experiment_parse_args_func)
