import json, os

config_path = 'config.json'
config_data = json.loads(open(config_path).read())  #, encoding='utf-8'

def load_param(path):
    try:
        param = json.loads(open(path).read())   #, encoding='utf-8'
    except:
        try:
            param = json.loads(open(path).read())   #, encoding='utf-8'
        except:
            try:
                param = json.loads(open(path).read())   #, encoding='utf-8'
            except:
                param = None

    return param
