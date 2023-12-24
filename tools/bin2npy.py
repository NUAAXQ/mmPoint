import os
from process_iwr1843 import RadarObject
import numpy as np

def bin2npy(bin_file, single_radarimg_dir):
    radarObject = RadarObject()
    radar_heatmaps = radarObject.processRadarDataHoriVert(bin_filename=bin_file)
    for indx, heatmap in enumerate(radar_heatmaps):

        radarpc_filename = single_radarimg_dir + '_' + str(indx).zfill(6) + '.npy'
        np.save(radarpc_filename, heatmap)
        print("%s scene %d frame has been saved!" % (single_radarimg_dir.split("/")[-1], indx))

if __name__ == "__main__":

    # need to be set manually
    radar_dir =  'your path to the raw radar signals from HuPR' # your path to the raw radar signals from HuPR
    radar_img_dir = 'your path to save the npy radar signal files' # your path to save the npy radar signal files

    # Create the folder if it doesn't exist
    if not os.path.exists(radar_img_dir):
        os.makedirs(radar_img_dir)

    # 58 scenes
    target_singles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15,
                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                      28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                      43, 100, 110, 120, 140, 150, 160, 170, 180, 190,
                      257, 259, 262, 265, 268, 270, 272, 273, 275, 276]

    single_radar_folders = sorted(os.listdir(radar_dir))
    for single_radar_folder in single_radar_folders:
        single_id = int(single_radar_folder.split('_')[-1])
        print('single_',single_id)
        if single_id not in target_singles:
            print(single_id,'continue')
            continue

        single_radar_dir = os.path.join(radar_dir,single_radar_folder) # 'radar/single_257'
        single_radarimg_dir = os.path.join(radar_img_dir,single_radar_dir.split('/')[-1]) #radar_img/single_257

        # use the vert radar in HuPR dataset
        bin_file = single_radar_dir + '/vert/adc_data.bin'

        bin2npy(bin_file, single_radarimg_dir)

        print("%s file saved successfully!"%(single_radar_folder))