import torch.nn as nn

class BasicBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, batchnorm=True,
                 activation=nn.ReLU):
        super(BasicBlock2D, self).__init__()
        if batchnorm:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                activation(),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                activation(),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            )
        self.relu = activation()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.main(x) + residual
        out = self.relu(out)
        return out

class Encoder2D_4layers(nn.Module):
    def __init__(self):
        super(Encoder2D_4layers, self).__init__()
        self.numFilters = 16  # cfg.MODEL.numFilters
        self.width = 64 # cfg.DATASET.heatmapSize
        self.height = 64 # cfg.DATASET.heatmapSize
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.numFilters, self.numFilters * 2, 3, 1, 1),
            BasicBlock2D(self.numFilters * 2, self.numFilters * 2, 3, 1, 1),
        )
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
            BasicBlock2D(self.numFilters * 2, self.numFilters * 4, 3, 1, 1),
            BasicBlock2D(self.numFilters * 4, self.numFilters * 4, 3, 1, 1),
        )
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
            BasicBlock2D(self.numFilters * 4, self.numFilters * 8, 3, 1, 1),
            BasicBlock2D(self.numFilters * 8, self.numFilters * 8, 3, 1, 1),
        )
        self.layer4 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
            BasicBlock2D(self.numFilters * 8, self.numFilters * 16, 3, 1, 1),
            BasicBlock2D(self.numFilters * 16, self.numFilters * 16, 3, 1, 1),
        )
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(16*self.numFilters, 1000)

    def forward(self, maps):
        '''
        new_maps : (batch_size, 32, 64, 64)
        '''
        x = self.layer1(maps)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x