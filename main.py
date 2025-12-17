import argparse
import os
import warnings

import numpy as np
import toml
import torch

import trainers
from utils.make_dir import make_dir

"""0. Setting the random seed"""
SEED = 2023
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
warnings.filterwarnings('ignore')


"""1. Reading configs from console"""
parser = argparse.ArgumentParser('Source-free_DA')
parser.add_argument('-c', '--config', 
                    default='./configs/A_source_only.toml',  # default
                    type=str, help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-Rx_s', default='1-1', type=str)
parser.add_argument('-Rx_t', default='1-19', type=str)
parser.add_argument('-save_dir', default='./saved/', type=str)


"""2. Loading toml configuration file"""
config, resume, Rx_s, Rx_t, save_dir = parser.parse_args().config, parser.parse_args().resume, parser.parse_args().Rx_s, parser.parse_args().Rx_t, parser.parse_args().save_dir
with open(config, 'r', encoding='utf8') as f:
    info = toml.load(f)


"""3. Start training ..."""
path = make_dir(info, config, Rx_s, Rx_t, save_dir)
print(f"--- Using configuration file: {config} ---")
print(f"--- Using device(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')} ---")

info['save_dir'] = save_dir
trainer = getattr(trainers, info['trainer'])(info, resume, path, Rx_s, Rx_t)  # load trainer from toml file
trainer.train()
