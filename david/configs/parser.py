import configparser
import os
from multiprocessing import cpu_count
from pathlib import Path

from .templates import INI_TEMPLATE_TENSORBOARD


def INIFileConfig(filename: str, template: str = None, exist_ok=False):
    """Create initialization files from templates or load config from file.

    `filename`: The name of the ini configuration file.
    `template`: The template is assumed to be a mapped str object.
    `exist_ok`: An error is raised if `exist_ok=False` and file exist.
    """
    filename = Path(filename)
    config = configparser.ConfigParser()
    config.optionxform = str

    def write(filename):
        with filename.open("w", encoding="utf8") as file:
            config.read_string(template)
            config.write(file)

    if template is not None:
        if not exist_ok:
            if not filename.exist():
                write(filename)
            else:
                raise FileExistsError(f"The file {filename} exists. Set "
                                     "exist_ok=True, to overwrite the file.")
        else:
            write(filename)
    else:
        template_lines = {}
        config.read(filename)
        for section in config.sections():
            template_lines[section] = {}
            if config.items(section) is None:
                continue
            for arg, val in config.items(section):
                template_lines[section][arg] = val
        return template


def cfg_tensorboard_ini(
    corpus_path, model_name=None, workers=None, root_dirname="models"
):
    """Save tensorboard's parameters to a config.ini from template."""
    if not workers:
        workers = cpu_count()

    corpus_path = os.path.realpath(corpus_path)
    abs_model_path = os.path.join(os.path.realpath(f"{root_dirname}/" + model_name))
    os.makedirs(abs_model_path, exist_ok=True)

    tb_ini_cfgs = INI_TEMPLATE_TENSORBOARD.format(
        corpus_path=corpus_path,
        model_name=model_name,
        model_path=abs_model_path,
        workers=workers,
    )
    ini_file_path = os.path.join(abs_model_path, f"{model_name}_model_.ini")
    return INIFileConfig(ini_file_path, default_ini_cfg=tb_ini_cfgs)
