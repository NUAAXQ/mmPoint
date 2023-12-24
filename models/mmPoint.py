from models.utils.decoder import *
import torch
import torch.nn as nn
from models.utils.HuPR_encoder import HuPRNet

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        # encoder
        self.encoder = HuPRNet()

        self.linear = nn.Linear(1000, 128)

        # decoder
        self.lift1 = Lift(args, up_factor=2,i=0)
        self.deform1 = Deform(args)

        self.lift2 = Lift(args, up_factor=2, i=0)
        self.deform2 = Deform(args)

        self.lift3 = Lift(args, up_factor=2, i=1)
        self.deform3 = Deform(args)

    def forward(self, images, sphere_points):
        sphere_points = torch.FloatTensor(sphere_points).unsqueeze(0).repeat(images.shape[0], 1, 1).cuda()
        sphere_points = sphere_points.transpose(2, 1).contiguous()

        mm_feat = self.encoder(images)
        mm_feat = self.linear(mm_feat)
        mm_feat = mm_feat.unsqueeze(-1)

        '''
        sphere_points: 256,3
        mm_feat: B,128
        '''

        '''
        step 1: 
        lift:   256,3  -> 512,3
        deform: 512,3 -> 512,3
        '''
        lift_point_cloud_1 = self.lift1(sphere_points,mm_feat)
        deform_point_cloud_1 = self.deform1(lift_point_cloud_1,mm_feat)

        '''
        step 2: 
        lift:   512,3  -> 1024,3
        deform: 1024,3 -> 1024,3
        '''
        lift_point_cloud_2 = self.lift2(deform_point_cloud_1, mm_feat)
        deform_point_cloud_2 = self.deform2(lift_point_cloud_2, mm_feat)

        '''
        step 3: 
        lift:   1024,3  -> 2048,3
        deform: 2048,3 -> 2048,3
        '''
        lift_point_cloud_3 = self.lift3(deform_point_cloud_2, mm_feat)
        deform_point_cloud_3 = self.deform3(lift_point_cloud_3, mm_feat)

        return deform_point_cloud_1.transpose(2, 1).contiguous(), \
            deform_point_cloud_2.transpose(2, 1).contiguous(), \
            deform_point_cloud_3.transpose(2, 1).contiguous(), \
            lift_point_cloud_1.transpose(2, 1).contiguous(), \
            lift_point_cloud_2.transpose(2, 1).contiguous(), \
            lift_point_cloud_3.transpose(2, 1).contiguous()