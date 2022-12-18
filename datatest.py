import os 
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile
from natsort import os_sorted

def create_dataloader():
    
        return DataLoader(
            dataset=Wave_Dataset('train'),
            batch_size=16,  # max 3696 * snr types
            shuffle=True, 
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
 
         # max 1152
def addr2wav(addr):
    wav, fs = soundfile.read(addr)
    # normalize
    wav = minMaxNorm(wav)
    return wav

def minMaxNorm(wav, eps=1e-8):
    max = np.max(abs(wav))
    min = np.min(abs(wav))
    wav = (wav - min) / (max - min + eps)
    return wav
def scan_directory(dir_name):
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()

    addrs = []
    for subdir, dirs, files in os.walk(dir_name, topdown=True):
        for file in files:
            if file.endswith(".wav"):
                filepath = subdir + file
                addrs.append(filepath)
    addrs=os_sorted(addrs)         
    return addrs

class Wave_Dataset(Dataset):
    def __init__(self, mode):
        # load data
        self.mode = mode
        pwd=os.getcwd()
        noisy_train=pwd+"/Webrtc_ns/battle_3"
        clean_train=pwd+"/Webrtc_ns/battle_3_1"    
        mode='train'
        
        if mode == 'train':
            print('<Training dataset>')
            print('Loading the data...')
            # load the wav addr
            self.noisy_dirs = scan_directory(noisy_train)
            #self.clean_dirs = scan_directory(cfg.clean_dirs)
            self.clean_dirs = scan_directory(clean_train)
           
       

    def __len__(self):
        return len(self.noisy_dirs)

    def __getitem__(self, idx):
        # read the wav
        inputs = addr2wav(self.noisy_dirs[idx])
        targets = addr2wav(self.clean_dirs[idx])

        return inputs, targets


class Bar(object):
    def __init__(self, dataloader):
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloder.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloder.')

        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        self._DISPLAY_LENGTH = 50

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())

        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)

        try:
            batch = next(self.iterator)
            self._display()
        except StopIteration:
            raise StopIteration()

        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._reset()

        return batch



    def _reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        
data= create_dataloader()
