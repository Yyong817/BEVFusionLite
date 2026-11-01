"""YAML inheritance and paths resolved from the project root."""
from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _merge(target, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def _expand(mapping, stack=()):
    result = {}
    if '_BASE_CONFIG_' in mapping:
        path = project_path(mapping['_BASE_CONFIG_'])
        if path in stack:
            raise ValueError(f'Circular config inheritance: {path}')
        with path.open(encoding='utf-8') as stream:
            result = _expand(yaml.safe_load(stream), (*stack, path))
    update = {key: _expand(value, stack) if isinstance(value, dict) else value
              for key, value in mapping.items() if key != '_BASE_CONFIG_'}
    return _merge(result, update)


def load_yaml(cfg_file):
    path = project_path(cfg_file)
    with path.open(encoding='utf-8') as stream:
        return _expand(yaml.safe_load(stream), (path,))


def cfg_from_yaml_file(cfg_file, config):
    from easydict import EasyDict
    config.update(EasyDict(load_yaml(cfg_file)))
    config.ROOT_DIR = ROOT
    config.LOCAL_RANK = 0
    if 'DATA_CONFIG' in config:
        config.DATA_CONFIG.DATA_PATH = str(project_path(config.DATA_CONFIG.DATA_PATH))
    image = config.get('MODEL', {}).get('IMAGE_BACKBONE', {})
    if image.get('INIT_CFG'):
        image.INIT_CFG.checkpoint = str(project_path(image.INIT_CFG.checkpoint))
    return config


def log_config_to_file(config, pre='cfg', logger=None):
    for key, value in config.items():
        if isinstance(value, dict):
            log_config_to_file(value, f'{pre}.{key}', logger)
        else:
            logger.info('%s.%s: %s', pre, key, value)


