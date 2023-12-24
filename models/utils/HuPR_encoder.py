import torch.nn as nn
from models.utils.chirp_networks import MNet
from models.utils.layers import Encoder2D_4layers

class HuPRNet(nn.Module):
    def __init__(self):
        super(HuPRNet, self).__init__()
        self.numFrames = 8 # cfg.DATASET.numFrames
        self.numFilters = 16  # cfg.MODEL.numFilters
        self.rangeSize = 64 # cfg.DATASET.rangeSize
        self.heatmapSize = 64 # cfg.DATASET.heatmapSize
        self.azimuthSize = 64 # cfg.DATASET.azimuthSize
        self.elevationSize = 8 # cfg.DATASET.elevationSize
        self.RAchirpNet = MNet(2, self.numFilters, self.numFrames)
        self.RAradarEncoder = Encoder2D_4layers()

    def forward_chirp(self, VRDAEmaps_hori):
        batchSize = VRDAEmaps_hori.size(0)
        VRDAmaps_hori = VRDAEmaps_hori.mean(dim=5)
        RAmaps = self.RAchirpNet(VRDAmaps_hori.view(batchSize, -1, self.numFrames, self.rangeSize, self.azimuthSize))
        RAmaps = RAmaps.squeeze(2).view(batchSize, -1, self.rangeSize, self.azimuthSize)

        return RAmaps

    def forward(self, VRDAEmaps_hori):
        '''
        VRDAEmaps_hori: batch_size,8,2,64,64,8
        '''
        RAmaps = self.forward_chirp(VRDAEmaps_hori)
        RAfeat = self.RAradarEncoder(RAmaps)
        return RAfeat

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)