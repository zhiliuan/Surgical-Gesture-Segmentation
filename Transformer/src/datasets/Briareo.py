import os
import math
import torch
from pathlib import Path

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset

from datasets.utils.normalize import normalize
from datasets.utils.optical_flow import dense_flow


##change in class __init__; train.json

class Briareo(Dataset):
    """Briareo Dataset class"""

    def __init__(self, npz, path, split, data_type='mix', transforms=None, n_frames=20,
                 optical_flow=False):
        """Constructor method for Briareo Dataset class

        Args:
            configer (Configer): Configer object for current procedure phase (train, test, val)
            split (str, optional): Current procedure phase (train, test, val)
            data_type (str, optional): Input data type (depth, rgb, normals, ir)
            transform (Object, optional): Data augmentation transformation for every data
            n_frames (int, optional): Number of frames selected for every input clip
            optical_flow (bool, optional): Flag to choose if calculate optical flow or not

        """
        super().__init__()

        self.dataset_path = Path(path)

        self.npz = npz
        self.split = split
        self.data_type = data_type
        self.optical_flow = optical_flow

        self.transforms = transforms
        self.n_frames = n_frames if not optical_flow else n_frames + 1

        print("Loading Briareo {} dataset...".format(split.upper()), end=" ")


        if self.split == 'test':
            data = None
            file_attrib = {}
            npz_path = str(self.dataset_path)+ '/'
            if data is None:
                data = np.load(npz_path + self.npz, allow_pickle=True)['arr_0']
                file_attrib[self.npz] = len(data)

            else:
                cur = np.load(npz_path + self.npz, allow_pickle=True)['arr_0']
                data = np.hstack((data, np.load(npz_path + self.npz, allow_pickle=True)['arr_0']))
                file_attrib[npz] = len(cur)

        elif self.split == 'train':
            npz_path = str(self.dataset_path) + '/' + self.split+ '/'
            npz_files = os.listdir(npz_path)
            # npz_files.sort()
            data = None
            file_attrib = {}
            for npz in npz_files:
                if data is None:
                    data = np.load(npz_path + npz, allow_pickle=True)['arr_0']
                    file_attrib[npz] = len(data)
                else:
                    cur = np.load(npz_path + npz, allow_pickle=True)['arr_0']
                    data = np.hstack((data, np.load(npz_path + npz, allow_pickle=True)['arr_0']))
                    file_attrib[npz] = len(cur)

        elif self.split == 'val':
            npz_path = str(self.dataset_path) + '/' + self.split+ '/'
            npz_files = os.listdir(npz_path)
            # npz_files.sort()
            data = None
            file_attrib = {}
            for npz in npz_files:
                if data is None:
                    data = np.load(npz_path + npz, allow_pickle=True)['arr_0']
                    file_attrib[npz] = len(data)
                else:
                    cur = np.load(npz_path + npz, allow_pickle=True)['arr_0']
                    data = np.hstack((data, np.load(npz_path + npz, allow_pickle=True)['arr_0']))
                    file_attrib[npz] = len(cur)


        self.data = data

        # Prepare clip for the selected number of frames n_frame
        # fixed_data = list()
        # for i, record in enumerate(data):
        #     paths = record['data']
        #     if len(paths) < 40:
        #         print('imagine number <= 40')
        #         print(record)
        #
        #     center_of_list = math.floor(len(paths) / 2)
        #     crop_limit = math.floor(self.n_frames / 2)
        #
        #     start = center_of_list - crop_limit
        #     end = center_of_list + crop_limit
        #     paths_cropped = paths[start: end + 1 if self.n_frames % 2 == 1 else end]
        #
        #     data[i]['data'] = paths_cropped
        #     fixed_data.append(data[i])
        #
        # self.data = np.array(fixed_data)
        print("done.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paths = self.data[idx]['data']
        # label = self.data[idx]['label']
        label = self.data[idx]['label']
        label = int(label)

        clip = list()
        for p in paths:
            img = cv2.imread(str(self.dataset_path / p), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            # img = cv2.resize(img, (56, 56))
            clip.append(img)

        clip = np.array(clip).transpose(1, 2, 3, 0)

        if self.optical_flow:
            clip = dense_flow(clip, self.data_type == "suturing")
        clip = normalize(clip)

        if self.transforms is not None:
            aug_det = self.transforms.to_deterministic()
            clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])]).transpose(1, 2, 3, 0)

        clip = torch.from_numpy(clip.reshape(clip.shape[0], clip.shape[1], -1).transpose(2, 0, 1))
        label = torch.LongTensor(np.asarray([label]))
        return clip.float(), label
