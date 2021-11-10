import pickle
import json
from torch import nn
from types import SimpleNamespace
from configparser import ConfigParser
from ast import literal_eval

def print_prediction(pred, k: int = 10, label_map_path: str = "imagenet_class_index.json"):
    with open(label_map_path) as f:
        imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}
    score = nn.Softmax()(pred)
    for i in range(k):
        max_ind = score.max(dim=1)[1].item()
        print(max_ind, imagenet_classes[max_ind], score[0][max_ind].item())
        score.data[0, max_ind] = 0

def object_save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        
def object_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def dict_eval(d):
    for k, v in d.items():
        try:    # number/tuple/dict/list type
            d[k] = literal_eval(v)
        except: # str type
            pass
    return d

def get_config(path): 
    cfg = ConfigParser()
    cfg.read(path, encoding='utf-8')
    config = {}
    for section in cfg.sections():
        config[section] = SimpleNamespace(**dict_eval(dict(cfg[section])))
    return SimpleNamespace(**config)
