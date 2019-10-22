import configparser
import os


def read_ini_config(config_path: str, default_ini_cfg: str):
    """Main ini configuration reader, writer and loader.

    Reads an ini file from the path, if the file exists, nothis is done, and
    simply returns a dictionary with its original settings (keys and values)
    Otherwise if the file does not exist - it creates a the file from the
    settings in the string object passed to the default_ini_cfg parameter.

    Parameters:
    ----------
    config_path : (str)
        The path to the ini file.
    default_ini_cfg : (str)
        The default configuartion settings to use in case
        a real ini file does not exist when reading the path.
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
