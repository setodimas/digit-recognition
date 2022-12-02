import yaml

yaml_file = 'configs/config.yaml'
        
# # Load configuration settings from yaml file
# with open(yaml_file, 'r') as config_file:
#     self.config = yaml.safe_load(config_file)
    
# return self.config

def load_config(path):
    with open(path, 'r') as file:
            config = yaml.safe_load(file)

    return config

if __name__ == "__main__":
    file_path = yaml_file
    config = load_config(file_path)
    print(config['dropout_probabilty'])