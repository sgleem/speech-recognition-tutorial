import json

def load_env(env_path="conf.json"):
    with open(env_path, 'r') as f:
        env=json.load(f)
    return env