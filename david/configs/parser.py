import configparser
import os
from multiprocessing import cpu_count
from .templates import INI_TEMPLATE_TENSORBOARD


def read_ini_config(config_path, default_ini_cfg):
    """Ini-file reader, writer and loader.

    Args:
        config_path (str): The path to the ini file.

        default_ini_cfg (str): The default configuartion settings
        to use in case a real ini file does not exist when reading the path.
    """
    config = configparser.ConfigParser()
    config.optionxform = str

    cfgs = {}
    if not os.path.exists(config_path):
        with open(config_path, 'w') as configfile:
            config.read_string(default_ini_cfg)
            config.write(configfile)
    else:
        config.read(config_path)

    for section in config.sections():
        cfgs[section] = {}
        if config.items(section) is None:
            continue
        for k, v in config.items(section):
            cfgs[section][k] = v
    return cfgs


def cfg_tensorboard_ini(corpus_path, model_name=None,
                        workers=None, root_dirname='models'):
    if not workers:
        workers = cpu_count()
    corpus_path = os.path.realpath(corpus_path)
    abs_model_path = os.path.join(
        os.path.realpath(f'{root_dirname}/' + model_name))
    os.makedirs(abs_model_path, exist_ok=True)

    # NOTE: make this method general so it can be used for other configs.
    tb_ini_cfgs = INI_TEMPLATE_TENSORBOARD.format(
        corpus_path=corpus_path,
        model_name=model_name,
        model_path=abs_model_path,
        workers=workers)

    ini_file_path = os.path.join(abs_model_path, f'{model_name}_model_.ini')
    return read_ini_config(ini_file_path, default_ini_cfg=tb_ini_cfgs)
