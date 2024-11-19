import yaml

def load_config(file_path:str) -> dict:
    with open(file_path, 'r') as file:
        config:dict = yaml.safe_load(file)
    return config

def load_config_section(file_path:str, section:str) -> dict:
    with open(file_path, 'r') as file:
        config:dict = yaml.safe_load(file)
    return config.get(section)

def print_config(config: dict):
    
    max_key_length = max(len(key)+1 for key in config.keys())
    for key, value in config.items():
        key = str(key)
        print(f"{key.ljust(max_key_length)}: {value}")
