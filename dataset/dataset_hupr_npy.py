import torch.utils.data as data
import os.path
import torch
import numpy as np
import os
import pickle
from os.path import join, dirname, exists
from easydict import EasyDict
from termcolor import colored
import dataset.pointcloud_processor as pointcloud_processor
from copy import deepcopy


def red_print(x):
    print(colored(x, "red"))

class HuPRDataset(data.Dataset):

    def __init__(self, opt, train=True):
        self.opt = opt
        self.train = train
        self.init_normalization()

        red_print('Create HuPR Dataset...')
        # Define core path array
        self.datapath = []

        # Load classes
        self.pointcloud_path = opt.pointcloud_path
        self.radar_path = opt.radar_path

        # Create Cache path
        self.path_dataset = opt.cache_path
        if not exists(self.path_dataset):
            os.mkdir(self.path_dataset)
        self.path_dataset = join(self.path_dataset,
                                 self.opt.normalization + str(train))

        # Compile list of pointcloud path by selected category
        dir_pointcloud = self.pointcloud_path
        dir_radar = self.radar_path
        list_pointcloud = sorted(os.listdir(dir_pointcloud))
        if self.train:
            list_pointcloud = list_pointcloud[:31800]
        else:
            list_pointcloud = list_pointcloud[31800:]
        print(' Number Files :'+ colored(str(len(list_pointcloud)), "yellow"))

        if len(list_pointcloud) != 0:
            for pointcloud in list_pointcloud:
                pointcloud_path = join(dir_pointcloud, pointcloud)
                radar_path = join(dir_radar, pointcloud)
                self.datapath.append((pointcloud_path, radar_path))

        # Preprocess and cache files
        self.preprocess()

    def preprocess(self):
        if exists(self.path_dataset + "info.pkl"):
            # Reload dataset
            red_print("Reload dataset")
            with open(self.path_dataset + "info.pkl", "rb") as fp:
                self.data_metadata = pickle.load(fp)

            self.data_points = torch.load(self.path_dataset + "points.pth")
        else:
            # Preprocess dataset and put in cache for future fast reload
            red_print("preprocess dataset...")
            self.datas = [self._getitem(i) for i in range(self.__len__())]

            # Concatenate all proccessed files
            self.data_points = [a[0] for a in self.datas]
            self.data_points = torch.cat(self.data_points, 0)

            self.data_metadata = [{'pointcloud_path': a[1], 'radar_path': a[2]} for a in self.datas]

            # Save in cache
            with open(self.path_dataset + "info.pkl", "wb") as fp:  # Pickling
                pickle.dump(self.data_metadata, fp)
            torch.save(self.data_points, self.path_dataset + "points.pth")

        red_print("Dataset Size: " + str(len(self.data_metadata)))

    def init_normalization(self):
        red_print("Dataset normalization : " + self.opt.normalization)

        if self.opt.normalization == "UnitBall":
            self.normalization_function = pointcloud_processor.Normalization.normalize_unitL2ball_functional
        elif self.opt.normalization == "BoundingBox":
            self.normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        else:
            self.normalization_function = pointcloud_processor.Normalization.identity_functional


    def _getitem(self, index):
        pointcloud_path, radar_path = self.datapath[index]
        points = np.load(pointcloud_path)
        points = torch.from_numpy(points).float()
        points[:, :3] = self.normalization_function(points[:, :3])
        return points.unsqueeze(0), pointcloud_path, radar_path

    def __getitem__(self, index):
        return_dict = deepcopy(self.data_metadata[index])

        # read points
        points = self.data_points[index]
        points = points.clone()
        return_dict['points'] = points[:, :3].contiguous()

        # read radar npy file
        im = torch.Tensor(np.load(return_dict['radar_path'], allow_pickle=True))
        return_dict['image'] = im
        return return_dict

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    print('Testing HuPR dataset')
    opt = {"normalization": "UnitBall", "class_choice": ["plane"], "SVR": True, "sample": True, "npoints": 2500}
    d = HuPRDataset(EasyDict(opt), train=False, keep_track=True)
    print(d[1])
    a = len(d)
