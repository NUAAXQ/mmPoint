import os
import numpy as np
import torch
import torchvision.transforms as transforms

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, radarData):
        c = radarData.size(0)
        minValues = torch.min(radarData.view(c, -1), 1)[0].view(c, 1, 1)
        radarDataZero = radarData - minValues
        maxValues = torch.max(radarDataZero.view(c, -1), 1)[0].view(c, 1, 1)
        radarDataNorm = radarDataZero / maxValues
        std, mean = torch.std_mean(radarDataNorm.view(c, -1), 1)
        return (radarDataNorm - mean.view(c, 1, 1)) / std.view(c, 1, 1)

# need to be set manually
source_dir = 'your path to save the npy radar signal files' # your path to save the npy radar signal files
target_dir = 'your path to save the final input hfd5 files' # your path to save the final input hfd5 files


if not os.path.exists(target_dir):
    os.makedirs(target_dir)

npy_files = sorted(os.listdir(source_dir),reverse=True)
print('files len:', len(npy_files))

radar_npy_transforms = transforms.Compose([
            transforms.ToTensor(),
            Normalize()
        ])

for npy_file in npy_files:
    npy_file_path = os.path.join(source_dir, npy_file)

    VRDAERealImag_hori = np.load(npy_file_path)

    VRDAEmaps_hori = torch.zeros((8, 2, 64, 64, 8))
    idxSampleChirps = 0
    numChirps = 16
    numFrames = 8
    for idxChirps in range(numChirps // 2 - numFrames // 2, numChirps // 2 + numFrames // 2):
        VRDAEmaps_hori[idxSampleChirps, 0, :, :, :] = radar_npy_transforms(
            VRDAERealImag_hori[idxChirps].real).permute(1, 2, 0)
        VRDAEmaps_hori[idxSampleChirps, 1, :, :, :] = radar_npy_transforms(
            VRDAERealImag_hori[idxChirps].imag).permute(1, 2, 0)
        idxSampleChirps += 1

    target_file_path = os.path.join(target_dir, npy_file)
    np.save(target_file_path, VRDAEmaps_hori.detach().cpu().numpy())
    print("%s frame has been saved!" % (target_file_path))