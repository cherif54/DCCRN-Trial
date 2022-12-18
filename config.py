"""
Configuration for program
"""

# model
mode = 'DCCRN'  # DCUNET / DCCRN
info = 'MODEL INFORMATION : IT IS USED FOR FILE NAME'

test = True

# path
job_dir = '/home/cherif/ARL/VBx/DCCRN-with-various-loss-functions/job/'
logs_dir = '/home/cherif/ARL/VBx/DCCRN-with-various-loss-functions/logs/'
#chkpt_path = '/home/cherif/ARL/VBx/DCCRN-with-various-loss-functions/job/12.9_DCCRN_MODEL INFORMATION : IT IS USED FOR FILE NAME/chkpt_10.pt'
chkpt_path='/home/cherif/ARL/VBx/DCCRN-with-various-loss-functions/job/first_model/chkpt_10.pt'
#chkpt_path= None

chkpt_model = 'first_model'
#chkpt_path = job_dir + chkpt_model + 'chkpt_10.pt'

# dataset path
wav_length=3
noisy_dirs_for_train = '/home/cherif/ARL/VBx/DCCRN-Datasets/train/noisy/'
clean_dirs_for_train = '/home/cherif/ARL/VBx/DCCRN-Datasets/train/clean/'
noisy_dirs_for_valid = '/home/cherif/ARL/VBx/DCCRN-Datasets/valid/noisy/'
clean_dirs_for_valid = '/home/cherif/ARL/VBx/DCCRN-Datasets/valid/clean/'

test_folder = '/home/cherif/ARL/VBx/DCCRN-Datasets/test/'
test_output = '/home/cherif/ARL/VBx/DCCRN-with-various-loss-functions/heli_3'

# model information
fs = 16000
FS= 16000
win_len = 400
win_inc = 100
ola_ratio = win_inc / win_len
fft_len = 512
sam_sec = fft_len / fs
frm_samp = fs * (fft_len / fs)
window_type = 'hann'

rnn_layers = 256
rnn_units = 64
masking_mode = 'E'
use_clstm = True
kernel_num = [32, 64, 128, 256, 256, 256]  # DCCRN
#kernel_num = [72, 72, 144, 144, 144, 160, 160, 180]  # DCUNET
loss_mode = 'MSE'

# hyperparameters for model train
max_epochs = 12
learning_rate = 0.0005
batch =22
