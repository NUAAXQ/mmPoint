import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class MNet(nn.Module):
    def __init__(self, in_channels, out_channels, numFrames):
        super(MNet, self).__init__()
        sizeTemp = sizeTempStride = numFrames//2
        self.temporalConvWx1x1 = nn.Conv3d(in_channels, out_channels, (2, 1, 1), (2, 1, 1), (0, 0, 0))
        self.temporalMaxpool = nn.MaxPool3d((sizeTemp, 1, 1), (sizeTempStride, 1, 1))
    def forward(self, chirpMaps):
        chirpMaps = self.temporalConvWx1x1(chirpMaps) #(batch_size,2,8,64,64) -> (batch_size,32,8,64,64)
        maps = self.temporalMaxpool(chirpMaps) # (batch_size,32,1,64,64)
        return maps