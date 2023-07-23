import argparse
import os

from train import GestureTrainer
from test import GestureTest
from predict import GesturePredict
from utils.configer import Configer


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
    parser.add_argument('--phase', default='test', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0, ], nargs='+', type=int,
                        dest='gpu', help='The gpu used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='resume', help='The path of pretrained model.')
    parser.add_argument('--nogesture', default=False, action='store_true',
                        dest='nogesture', help='NoGesture CTC loss')

    args = parser.parse_args()
    args.hypes = '/content/Transformer_mix/src/hyperparameters/Briareo/train.json'
    resumes_model_path = '/content/train_briareo_80.pth'

    args.device = None
    # args.phase = 'test'
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    torch.autograd.set_detect_anomaly(True)

    # args.phase = 'test'
    configer = Configer(args)
    if configer.get('phase') == 'train':
        args.resume = ''
        model = GestureTrainer(configer)
        model.init_model()
        features = model.train()


    elif configer.get('phase') == 'test':
        train_dataset_path = '/content/npz_files/'
        npz_path = train_dataset_path
        npz_files = os.listdir(npz_path)
        for npz in npz_files[36:]:
            print(npz)
            args.resume = resumes_model_path
            model = GestureTest(configer, npz)
            model.init_model()
            features, gt, predict = model.test()
            np.savetxt(npz[:-4] + ".txt", features, fmt='%f', delimiter=' ')
            print('saving features file')
            prediction = np.hstack((gt, predict))
            np.savetxt('prediction' + npz[:-4] + ".txt", prediction, fmt='%f', delimiter=' ')





