import os
import shutil
import time
from pathlib import Path


def make_dir(info, config, Rx_s, Rx_t, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = Path(save_dir) / (Rx_s + '_' + Rx_t) # saved
    # if nTx > 0:
    #     if not os.path.exists(f'saved/wisig_manyTx/{nTx}Tx'):
    #         os.mkdir(f'saved/wisig_manyTx/{nTx}Tx')
    #     save_dir = Path(f'saved/wisig_manyTx/{nTx}Tx') / (Rx_s + '_' + Rx_t) # saved

    # save_dir = Path('saved/wisig_snr') / (Rx_s + '_' + Rx_t)
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # if not os.path.exists(save_dir / f"{int(snr)}dB"):
    #     os.mkdir(save_dir / f"{int(snr)}dB")
    # save_dir = save_dir / f"{int(snr)}dB"


    single_dir = save_dir / (Rx_s + '_' + Rx_t + '_' + info['name'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(single_dir):
        os.mkdir(single_dir)
    time_list = list(time.localtime(time.time()))
    time_string = '-'.join(map(lambda x: str(x).zfill(2), time_list[:6]))
    if not os.path.exists(single_dir / time_string):
        os.mkdir(single_dir / time_string)
        os.mkdir(single_dir / time_string / 'log')
        os.mkdir(single_dir / time_string / 'model')
        os.mkdir(single_dir / time_string / 'confusion_matrix')
    shutil.copy(config, single_dir / time_string / 'log')
    return single_dir / time_string
