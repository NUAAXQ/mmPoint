import torch
import sys
import os

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/ChamferDistancePytorch"))
from chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()


def calc_cd(output, gt):
    cham_loss = chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    return cd_p, cd_t

def get_cd_loss(deform_points_1, deform_points_2, deform_points_3, lift_points_1, lift_points_2, lift_points_3, gt):
    # reconstruction loss
    cd_loss_d_1, loss_t_d_1 = calc_cd(deform_points_1, gt)
    cd_loss_d_1 = cd_loss_d_1.mean()
    cd_loss_l_1, loss_t_l_1 = calc_cd(lift_points_1, gt)
    cd_loss_l_1 = cd_loss_l_1.mean()

    cd_loss_d_2, loss_t_d_2 = calc_cd(deform_points_2, gt)
    cd_loss_d_2 = cd_loss_d_2.mean()
    cd_loss_l_2, loss_t_l_2 = calc_cd(lift_points_2, gt)
    cd_loss_l_2 = cd_loss_l_2.mean()

    cd_loss_d_3, loss_t_d_3 = calc_cd(deform_points_3, gt)
    cd_loss_d_3 = cd_loss_d_3.mean()
    cd_loss_l_3, loss_t_l_3 = calc_cd(lift_points_3, gt)
    cd_loss_l_3 = cd_loss_l_3.mean()

    cd_loss_l = cd_loss_l_1 + cd_loss_l_2 + cd_loss_l_3
    cd_loss_d = cd_loss_d_1 + cd_loss_d_2 + cd_loss_d_3

    cd_loss = cd_loss_l + cd_loss_d

    return cd_loss, loss_t_d_3, cd_loss_d_3


