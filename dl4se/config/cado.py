
import dl4se.config as config
import dl4se.util as util


def set_defaults(config):
    config.train_epochs = 3
    config.lr = 1e-5
    # config.model_path = util.models_path('android.stackexchange.com')
    # config.model_path = util.models_path('android.stackexchange.com')#a
    # config.model_path = util.models_path('stackoverflow_1M')
    # config.model_path = 'bert-base-uncased'
    # config.model_path = util.models_path('StackOBERTflow-comments-small-v1/')
    # config.model_path = 'bert-large-uncased'
    config.max_seq_len = 255
    config.model_type = None
    config.model_path = util.models_path('StackOBERTflow-comments-small-v1')
    config.train_bs = 32
    config.eval_bs = 256
    #config.hidden_dropout_prob = 0.07
    config.num_labels = 12
    config.hidden_dropout_prob = 0.1
    config.multi_label = True

    #config.label_weights = None
    # sorted by label name (canonical class order)
    # config.label_weights = [0.1225, 0.5,
    #                             0.1225, 0.1225,
    #                             0.1225, 0.01]
    # config.label_weights = [0.0089912, 0.07057243, 0.05001939, 0.05292932, 0.23056881,
    #                         0.08698462, 0.02226392, 0.04400398, 0.04095261, 0.30883529,
    #                         0.0658768, 0.01800163]


def parse_args(parser):
    parser.add_argument("--single_class", default=None,
                        type=str, required=False)

def after_parse_func(config):
    if config.single_class:
        config.multi_label = False

def get_config(experiment_parse_args_func=None):
    return config.get_config(set_defaults, parse_args, experiment_parse_args_func=experiment_parse_args_func, after_parse_func=after_parse_func)
