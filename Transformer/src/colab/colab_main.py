import argparse
import os

from train import GestureTrainer
from test import GestureTest
from utils.configer import Configer
from features import get_attribute

import random
import numpy as np
import torch

SEED = 1994
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True  # To have ~deterministic results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--hypes', default=None, type=str,
                        dest='hypes', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0, ], nargs='+', type=int,
                        dest='gpu', help='The gpu used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='resume', help='The path of pretrained model.')
    parser.add_argument('--nogesture', default=False, action='store_true',
                        dest='nogesture', help='NoGesture CTC loss')
    args = parser.parse_args()

    args.hypes = '/content/Transformer_Needle/src/hyperparameters/Briareo/train.json'
    # resumes = ['best_train_frame20.pth','best_train_frame30.pth','Maybeold.pth']
    resumes = ['step5_len20_41.pth']

    # args.resume =/content/best_train_briareo (1).pth
    resumes_model_path = '/content/model/'

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    torch.autograd.set_detect_anomaly(True)

    args.phase = 'test'
    configer = Configer(args)

    if configer.get('phase') == 'train':
        model = GestureTrainer(configer)
        model.init_model()
        features = model.train()
    elif configer.get('phase') == 'test':
        train_dataset_path = '/content/Transformer_Needle/needle/train_step1/'
        npz_path = train_dataset_path + 'test' + '/'
        npz_files = os.listdir(npz_path)
        for npz in npz_files:
            for r in resumes:
                print(r)
                print(npz)
                args.resume = resumes_model_path + r
                model = GestureTest(configer, npz)
                model.init_model()
                features = model.test()
                print('saving features file')
                np.savetxt(npz[:-4]+ ".txt", features, fmt='%f', delimiter=' ')
    # file_attrib = get_attribute('train')
    # i = 0
    # for key, length in file_attrib.items():
    #     write_in = features[i:int((i + length)/3)]
    #     i += length
    #     np.savetxt(key[15:-4] + ".txt", write_in, fmt='%f', delimiter=' ')



