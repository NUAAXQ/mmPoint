import warnings
import numpy as np
import torch
import random
import importlib
import munch
import yaml
import argparse
from utils.model_utils import *

warnings.filterwarnings("ignore")

def predict_one_pc(input_radar_path, load_model):
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    model_name = 'mmPoint'
    model_module = importlib.import_module('.%s' % model_name, 'models')

    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()

    ckpt = torch.load(load_model)
    net.module.load_state_dict(ckpt['net_state_dict'])

    net.module.eval()

    radar = torch.Tensor(np.load(input_radar_path, allow_pickle=True))
    radar = radar.unsqueeze(0).cuda()

    template_name = '../human_template/human_template_256.xyz'
    template_points = np.loadtxt(template_name)
    template_points = pc_normalize(template_points, 0.5)

    _, _, deformed_points_3, _, _, _ = net(radar, template_points)

    return deformed_points_3.squeeze().cpu().detach().numpy()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', default='../cfgs/mmPoint.yaml')
    parser.add_argument('-gpu', '--gpu_id', help='gpu_id', default=0)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    # need to be set manually
    input_npy_path = 'put your input radar npy file here' # put your input radar npy file here

    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)

    # need to be set manually
    model_name = 'trained model .pth file'


    predict_pc = predict_one_pc(input_npy_path, model_name)
    savetxt = save_dir + '/test.txt'
    np.savetxt(savetxt, predict_pc, delimiter=';')
    print('%s point cloud saved successfully!'%save_dir)