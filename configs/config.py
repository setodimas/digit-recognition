import yaml
import os


def load_config(path):
    with open(path, 'r') as file:
            config = yaml.safe_load(file)

    return config

if __name__ == "__main__":
    file_path = 'configs/config.yaml'
    config = load_config(file_path)
    print(config)