import argparse
import yaml

yaml_file = 'configs/config.yaml'

class _Config:
    def __init__(self):
        print('Loading config')
        with open(yaml_file, 'r') as yaml_config_file:
            self.config = yaml.safe_load(yaml_config_file)
        parser = argparse.ArgumentParser()
        parser.add_argument('--an_arg')
        self.args = parser.parse_args()

    def __getattr__(self, name):
        try:
            return self.config[name]
        except KeyError:
            return getattr(self.args, name)
        
config = _Config()
