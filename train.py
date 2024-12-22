from utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, write_2images, Timer
import argparse
from trainer import HIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/imagenet.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True  # TODO QIAO

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']

# qss, ok, display: images, hierarchical labels debug
train_loader, test_loader = get_all_data_loaders(config)

train_samples, test_samples, train_h0_labels, test_h0_labels, train_h1_labels, test_h1_labels, train_h2_labels, \
test_h2_labels, train_h0_local_labels, test_h0_local_labels, train_h1_local_labels, test_h1_local_labels = \
 list([]), list([]), list([]), list([]), list([]), list([]), list([]), list([]), list([]),list([]), list([]), list([])

for i in range(display_size):
    train_one_sample = train_loader.dataset[i]
    test_one_sample = test_loader.dataset[i]
    train_samples.append(train_one_sample[0])
    test_samples.append(test_one_sample[0])
    train_h0_labels.append(train_one_sample[1])
    test_h0_labels.append(test_one_sample[1])
    train_h1_labels.append(train_one_sample[2])
    test_h1_labels.append(test_one_sample[2])
    train_h2_labels.append(train_one_sample[3])
    test_h2_labels.append(test_one_sample[3])
    train_h0_local_labels.append(train_one_sample[4])
    test_h0_local_labels.append(test_one_sample[4])
    train_h1_local_labels.append(train_one_sample[5])
    test_h1_local_labels.append(test_one_sample[5])

train_display_data = torch.stack(train_samples).cuda()
train_display_h0_labels = torch.Tensor(train_h0_labels).cuda()
train_display_h1_labels = torch.Tensor(train_h1_labels).cuda()
train_display_h2_labels = torch.Tensor(train_h2_labels).cuda()
train_display_h0_local_labels = torch.Tensor(train_h0_local_labels).cuda()
train_display_h1_local_labels = torch.Tensor(train_h1_local_labels).cuda()
test_display_data = torch.stack(test_samples).cuda()
test_display_h0_labels = torch.Tensor(test_h0_labels).cuda()
test_display_h1_labels = torch.Tensor(test_h1_labels).cuda()
test_display_h2_labels = torch.Tensor(test_h2_labels).cuda()
test_display_h0_local_labels = torch.Tensor(test_h0_local_labels).cuda()
test_display_h1_local_labels = torch.Tensor(test_h1_local_labels).cuda()

# qss, Setup model and data loader, ok
trainer = HIT_Trainer(config)

trainer.cuda()

# Setup logger and output folders, ok
# model_name = os.path.splitext(os.path.basename(opts.config))[0]
model_name = config['model_name']
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# qss, Start training
initial_lambda_ds = config['ds_w']
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images, h0_labels, h1_labels, h2_labels, h0_local_labels, h1_local_labels) in enumerate(train_loader):
        # Main training code, debug
        trainer.update_learning_rate()
        images = images.cuda().detach()
        h0_labels = h0_labels.cuda().detach()
        h1_labels = h1_labels.cuda().detach()
        h2_labels = h2_labels.cuda().detach()
        h0_local_labels = h0_local_labels.cuda().detach()
        h1_local_labels = h1_local_labels.cuda().detach()
        with Timer("Elapsed time in update: %f"):
            trainer.dis_update(images, config, h0_labels, h0_local_labels, h1_labels, h1_local_labels, h2_labels)
            trainer.gen_update(images, config, h0_labels, h0_local_labels, h1_labels, h1_local_labels, h2_labels)
            torch.cuda.synchronize()

        # Dump training stats in log file, ok
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images, modify to loop, label problem debug
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():  # close the grad compute when test
                te_img_recon, te_img_trans, te_tar_lev, te_tar_labs, te_per_num = trainer.rand_trans(test_display_data)
                tr_img_recon, tr_img_trans, tr_tar_lev, tr_tar_labs, tr_per_num = trainer.rand_trans(train_display_data)
            write_2images(test_display_data, te_img_recon, te_img_trans, te_per_num, test_display_h0_labels, te_tar_lev,
                          te_tar_labs, image_directory, 'test_%08d' % (iterations + 1), config['w_scale'], config['h_scale'])
            write_2images(train_display_data, tr_img_recon, tr_img_trans, tr_per_num, train_display_h0_labels, tr_tar_lev,
                          tr_tar_labs, image_directory, 'train_%08d' % (iterations + 1), config['w_scale'], config['h_scale'])

        # Save network weights, ok
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        config['ds_w'] = initial_lambda_ds - iterations * (float(initial_lambda_ds) / config['step_size'])
        if iterations >= max_iter:
            sys.exit('Finish training')

