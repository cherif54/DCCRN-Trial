import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config as cfg
from tools import scan_directory, find_pair, addr2wav 
import os

# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def create_dataloader(mode):
    if mode == 'train':
        return DataLoader(
            dataset=Wave_Dataset(mode),
            batch_size=cfg.batch,  # max 3696 * snr types
            shuffle=True, 
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
    elif mode == 'valid':
        return DataLoader(
            dataset=Wave_Dataset(mode),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )    # max 1152


def create_dataloader_for_test(mode, type, snr):
    if mode == 'test':
        return DataLoader(
            dataset=Wave_Dataset_for_test(mode, type, snr),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )    # max 192

def create_dataloader_for_infer(mode):
    if mode == 'infer':
        return DataLoader(
            dataset=Wave_Dataset_for_infer(mode),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )    # max 192

class Wave_Dataset(Dataset):
    def __init__(self, mode):
        # load data
        self.mode = mode
        pwd=os.getcwd()
        noisy_train=pwd+"/noisy_train/"
        clean_train=pwd+"/clean_train/"    
        
        noisy_valid=pwd+"/noisy_valid/"
        clean_valid=pwd+"/clean_valid/"
        
        if mode == 'train':
            print('<Training dataset>')
            print('Loading the data...')
            # load the wav addr
            self.noisy_dirs = scan_directory(noisy_train)
            #self.clean_dirs = scan_directory(cfg.clean_dirs)
            self.clean_dirs = scan_directory(clean_train)
           
        elif mode == 'valid':
            print('<Validation dataset>')
            print('Loading the data...')
            # load the wav addr
            self.noisy_dirs = scan_directory(noisy_valid)
            #self.clean_dirs = find_pair(self.noisy_dirs)
            self.clean_dirs = scan_directory(clean_valid)

    def __len__(self):
        return len(self.noisy_dirs)

    def __getitem__(self, idx):
        # read the wav
        inputs = addr2wav(self.noisy_dirs[idx])
        targets = addr2wav(self.clean_dirs[idx])

        #with open("temporary_file_length.txt") as f:
        #        file_size = f.read() ##Assume the sample file has 3 lines

        #file_size=int(file_size)

        #inputs.resize(file_size)
        #targets.resize(file_size)


        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)
        #print("running ",self.noisy_dirs[idx],inputs.size()," and",self.clean_dirs[idx],targets.size())

        # (-1, 1)
        inputs = torch.clamp_(inputs, -1, 1)
        targets = torch.clamp_(targets, -1, 1)
        #print(self.noisy_dirs[idx],self.clean_dirs[idx],"Inputs size",inputs.size(),"Targets size", targets.size())
        return inputs, targets

class Wave_Dataset_for_test(Dataset):
    def __init__(self, mode, type, snr):
        # load data
        if mode == 'test':
            print('<Test dataset>')
            print('Load the data...')
            self.input_path = './input/recon_test_dataset.npy'

        self.input = np.load(self.input_path)
        self.input = self.input[type][snr]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        inputs = self.input[idx][0]
        

        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        

        return inputs, labels



class Wave_Dataset_for_infer(Dataset):
    def __init__(self, mode):
        # load data
        pwd=os.getcwd()
        test_loc=pwd+"/test_data/"
        
        print('<Testing dataset>')
        print('Loading the data...')
            # load the wav addr
        self.test_dirs = scan_directory(test_loc)

           
    def __len__(self):
        return len(self.test_dirs)

    def __getitem__(self, idx):
        # read the wav
        inputs = addr2wav(self.test_dirs[idx])


        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        # (-1, 1)
        inputs = torch.clamp_(inputs, -1, 1)

        return inputs
