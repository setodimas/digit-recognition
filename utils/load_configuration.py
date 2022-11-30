import yaml


class ConfigurationParametes:
    
    def __init__(self) -> None:
        """
        Initialize configurations file

        """
        
        yaml_file = './configs/config.yaml'
        
        # Load configuration settings from yaml file
        with open(yaml_file, 'r') as config_file:
            self.config_dict = yaml.safe_load(config_file)
            
        return