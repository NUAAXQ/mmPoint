import argparse
import torch
import munch
import dataset.dataset_hupr_npy as dataset_hupr
import yaml

def build_dataset(args):
    # Create Datasets
    dataset_train = dataset_hupr.HuPRDataset(args, train=True)
    dataset_test = dataset_hupr.HuPRDataset(args, train=False)

    # Create dataloaders
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                                    batch_size=args.batch_size,
                                                                    shuffle=True,
                                                                    pin_memory=True,
                                                                    prefetch_factor=4,
                                                                    persistent_workers=True,
                                                                    num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=args.batch_size,
                                                                shuffle=False,
                                                                pin_memory=True,
                                                                prefetch_factor=4,
                                                                persistent_workers=True,
                                                                num_workers=int(args.workers))

    len_dataset = len(dataset_train)
    len_dataset_test = len(dataset_test)
    print('Length of train dataset:%d', len_dataset)
    print('Length of test dataset:%d', len_dataset_test)

    return dataloader_train, dataloader_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-gpu', '--gpu_id', help='gpu_id', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    dataloader_train, dataloader_test = build_dataset(args)
