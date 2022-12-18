import os
import time
import torch
import numpy as np
import config as cfg
from train import model_test, model_infer
from dataloader import create_dataloader_for_test, create_dataloader_for_infer
from model import DCCRN

print("Running tester")
###############################################################################
#                        Helper function definition                           #
###############################################################################
# Write training related parameters into the log file.
def write_status_to_log_file(fp, total_parameters):
    fp.write('adsfasdfsdfds')
    fp.write('%d-%d-%d %d:%d:%d\n' %
             (time.localtime().tm_year, time.localtime().tm_mon,
              time.localtime().tm_mday, time.localtime().tm_hour,
              time.localtime().tm_min, time.localtime().tm_sec))
    fp.write('mode                : %s_%s\n' % (cfg.mode, cfg.info))
    fp.write('learning rate       : %g\n' % cfg.learning_rate)
    fp.write('total params   : %d (%.2f M, %.2f MBytes)\n' %
             (total_parameters,
              total_parameters / 1000000.0,
              total_parameters * 4.0 / 1000000.0))


# Calculate the size of total network.
def calculate_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


###############################################################################
#                          Parameter Initialization                           #
###############################################################################
print('***********************************************************')
print('*    Python library for DNN-based speech enhancement      *')
print('*                        using Pytorch API                *')
print('***********************************************************')

# Set device
DEVICE = torch.device("cuda")
truncator(cfg.test_folder,"test_data")

print("Ran test_data")
# Set model
if cfg.mode == 'DCCRN':
    model = DCCRN(rnn_units=cfg.rnn_units, masking_mode=cfg.masking_mode, use_clstm=cfg.use_clstm,
                  kernel_num=cfg.kernel_num).to(DEVICE)

###############################################################################
#                    Set optimizer and learning rate                          #
###############################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
total_params = calculate_total_params(model)

###############################################################################
#                          Confirm model information                          #
###############################################################################
print('%d-%d-%d %d:%d:%d\n' %
      (time.localtime().tm_year, time.localtime().tm_mon,
       time.localtime().tm_mday, time.localtime().tm_hour,
       time.localtime().tm_min, time.localtime().tm_sec))
print('mode                : %s_%s\n' % (cfg.mode, cfg.info))
print('learning rate       : %g\n' % cfg.learning_rate)
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))


###############################################################################
#                        Set a log file to store progress.                    #
#               Set a hps file to store hyper-parameters information.         #
###############################################################################
# Load the checkpoint
if cfg.chkpt_path is not None:
    print('Resuming from checkpoint: %s' % cfg.chkpt_path)

    # Set a log file to store progress.
    dir_to_save = cfg.job_dir + cfg.chkpt_model
    dir_to_logs = cfg.logs_dir + cfg.chkpt_model

    checkpoint = torch.load(cfg.chkpt_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start_idx = checkpoint['epoch'] + 1
    mse_vali_total = np.load(str(dir_to_save + '/mse_vali_total.npy'))
    if len(mse_vali_total) < cfg.max_epochs:
        plus = cfg.max_epochs - len(mse_vali_total)
        mse_vali_total = np.concatenate((mse_vali_total, np.zeros(plus)), 0)


if not os.path.exists(dir_to_save):
    os.mkdir(dir_to_save)
    os.mkdir(dir_to_logs)

log_fname = str(dir_to_save + '/log.txt')
if not os.path.exists(log_fname):
    fp = open(log_fname, 'w')
    write_status_to_log_file(fp, total_params)
else:
    fp = open(log_fname, 'a')

# Set a hps file to store hyper-parameters information.
hps_fname = str(dir_to_save + '/hp_str.txt')
fp_h = open(hps_fname, 'w')

with open('config.py', 'r') as f:
    hp_str = ''.join(f.readlines())
fp_h.write(hp_str)
fp_h.close()

min_index = np.argmin(mse_vali_total)
print('Minimum validation loss is at '+str(min_index+1)+'.')

###############################################################################
#                                    Test                                     #
###############################################################################
if cfg.test is True:
    print('Starting test run')

    # check the lowest validation loss epoch
    want_to_check = torch.load(dir_to_save + '/chkpt_opt.pt')
    model.load_state_dict(want_to_check['model'])
    optimizer.load_state_dict(want_to_check['optimizer'])
    epoch_start_idx = want_to_check['epoch'] + 1
    mse_vali_total = np.load(str(dir_to_save + '/mse_vali_total.npy'))

    test_loader = create_dataloader_for_infer(mode)
    model_infer(model, test_loader, DEVICE)

    fp.close()
else:
    fp.close()
