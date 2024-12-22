from __future__ import print_function
from utils import get_config, __write_images
from trainer import HIT_Trainer
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import numpy as np
import utils as utl
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import torch
import os, sys

#
# Main features: note the input value range for different features ([-1,1], [0,255] or any others)
# 1. random sample num (e.g. 3 or 5) styles from each tree node for each input image
# 2. test LPIPS for random selected 100 inputs with 19 sampling styles for each in the intermediate level
# 3. test IS and FID for random selected 100 inputs with 100 sampling styles for each in the intermediate level
# LPIPS, FID and IS are tested in independent project (perceptive similarity) or codes (FID, IS)
#

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/imagenet.yaml', help='Path to the config file.')
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether output raw input images")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
np.random.seed(opts.seed)
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
# Setup model and data loader
trainer = HIT_Trainer(config)

total_params = sum(p.numel() for p in trainer.parameters())
print('total_params: {}'.format(total_params / 1e6))

root_num = trainer.gen.distributions.root_num
root_tar = np.random.randint(low=0, high=root_num)
inter_num = trainer.gen.distributions.intermediate_num
inter_tar = np.random.randint(low=0, high=inter_num)
leaf_num = trainer.gen.distributions.leaves_num
leaf_tar = np.random.randint(low=0, high=leaf_num)
total_num = root_num + inter_num + leaf_num

_, test_loader = utl.get_all_data_loaders(config)

model_name = config['model_name']
input_directory = os.path.join("./results", model_name, 'inputs/')
output_directory1 = os.path.join("./results", model_name, 'translation_display/')
output_directory2 = os.path.join("./results", model_name, 'translation_realism/')
output_directory3 = os.path.join("./results", model_name, 'translation_diversity/')

if not os.path.exists(input_directory):
    print("Creating directory: {}".format(input_directory))
    os.makedirs(input_directory)
if not os.path.exists(output_directory1):
    print("Creating directory: {}".format(output_directory1))
    os.makedirs(output_directory1)
    os.makedirs(os.path.join(output_directory1, 'level0'))
    os.makedirs(os.path.join(output_directory1, 'level1'))
    os.makedirs(os.path.join(output_directory1, 'level2'))

if not os.path.exists(output_directory2):
    print("Creating directory: {}".format(output_directory2))
    os.makedirs(output_directory2)
    os.makedirs(os.path.join(output_directory2, 'level0'))
    os.makedirs(os.path.join(output_directory2, 'level1'))
    os.makedirs(os.path.join(output_directory2, 'level2'))
if not os.path.exists(output_directory3):
    print("Creating directory: {}".format(output_directory3))
    os.makedirs(output_directory3)
    os.makedirs(os.path.join(output_directory3, 'level0'))
    os.makedirs(os.path.join(output_directory3, 'level1'))
    os.makedirs(os.path.join(output_directory3, 'level2'))


state_dict = torch.load(opts.checkpoint)
trainer.gen.load_state_dict(state_dict['gen'])


trainer.cuda()
trainer.eval()
encode = trainer.gen.encode
decode = trainer.gen.decode

# Start testing
display_style_num = 5
test_img_num = 100
lpips_style_num = 38
fid_is_style_num = 100
max_test_num = 500
count = 0
sample_num = test_loader.dataset.__len__()

with torch.no_grad():
    style_fixed = trainer.gen.distributions.forward(sample_size=fid_is_style_num)
    for _, (images, h0_labels, h1_labels, h2_labels, _, _) in enumerate(test_loader):
        images = Variable(images.cuda())
        content, style_real = encode(images)
        batch_size = images.size(0)
        for i in range(batch_size):
            if count >= max_test_num or count >= sample_num:
                sys.exit('Finish test')
            count = count + 1
            if not opts.output_only:
                vutils.save_image(images[i].data, os.path.join(input_directory, '{:03d}_{:02d}_{:02d}_{:02d}.jpg'.
                                                               format(count, h0_labels[i], h1_labels[i], h2_labels[i])),
                                  padding=0, normalize=True)
            c1 = content[i].unsqueeze(0)
            s1 = style_real[i].unsqueeze(0)
            if opts.synchronized:
                style = style_fixed
            else:
                style = trainer.gen.distributions.forward(sample_size=fid_is_style_num)

            # display translation for each node
            for ndx in range(total_num):
                if ndx < leaf_num:
                    level = 2
                    src_lab = h0_labels[i]
                    target_lab = ndx

                elif leaf_num <= ndx < leaf_num + inter_num:
                    level = 1
                    src_lab = h1_labels[i]
                    target_lab = ndx - leaf_num

                else:
                    level = 0
                    src_lab = h2_labels[i]
                    target_lab = ndx - leaf_num - inter_num

                s_disp = style[ndx*fid_is_style_num:ndx*fid_is_style_num+display_style_num]
                c_disp = c1.repeat(display_style_num, 1, 1, 1)


                # translation display using sampled styles
                x_disp = decode(c_disp, s_disp)
                x_disp = torch.cat([images[i].unsqueeze(0), x_disp])
                __write_images(x_disp, '%s/%03d_%02dTo%02d.jpg' % ((output_directory1+'level'+str(level)+'/'), count,
                                                                   src_lab, target_lab),
                               config['w_scale'], config['h_scale'])

            # continue, if don't want to repeat the following TODO
            # continue
            # for quantitative test
            if count <= test_img_num:

                leaf_tar = np.random.randint(low=0, high=leaf_num)
                inter_tar = np.random.randint(low=0, high=inter_num)
                root_tar = np.random.randint(low=0, high=root_num)
                c_fid_is = c1.repeat(fid_is_style_num, 1, 1, 1)
                s_fid_is_l2 = style[leaf_tar * fid_is_style_num:leaf_tar * fid_is_style_num + fid_is_style_num]
                s_fid_is_l1 = style[(leaf_num + inter_tar) * fid_is_style_num:
                                    (leaf_num + inter_tar) * fid_is_style_num + fid_is_style_num]
                s_fid_is_l0 = style[(leaf_num + inter_num + root_tar) * fid_is_style_num:
                                    (leaf_num + inter_num + root_tar) * fid_is_style_num + fid_is_style_num]
                x_fid_is_l2 = decode(c_fid_is, s_fid_is_l2)
                x_fid_is_l1 = decode(c_fid_is, s_fid_is_l1)
                x_fid_is_l0 = decode(c_fid_is, s_fid_is_l0)
                for idx_fid_is, (img_l2, img_l1, img_l0) in enumerate(zip(x_fid_is_l2, x_fid_is_l1, x_fid_is_l0)):
                    if idx_fid_is < lpips_style_num:
                        vutils.save_image(x_fid_is_l2[idx_fid_is].data, os.path.join(output_directory3,
                                          'level2/{:03d}_{:02d}To{:02d}_{:02d}.jpg'.format(
                                              count, h0_labels[i], leaf_tar, idx_fid_is)),
                                          padding=0, normalize=True)
                        vutils.save_image(x_fid_is_l1[idx_fid_is].data, os.path.join(output_directory3,
                                          'level1/{:03d}_{:02d}To{:02d}_{:02d}.jpg'.format(
                                              count, h1_labels[i], inter_tar, idx_fid_is)),
                                          padding=0, normalize=True)
                        vutils.save_image(x_fid_is_l0[idx_fid_is].data, os.path.join(output_directory3,
                                          'level0/{:03d}_{:02d}To{:02d}_{:02d}.jpg'.format(
                                              count, h2_labels[i], root_tar, idx_fid_is)),
                                          padding=0, normalize=True)
                    vutils.save_image(img_l2.data, os.path.join(output_directory2,
                                      'level2/{:03d}_{:02d}To{:02d}_{:03d}.jpg'.format(count, h0_labels[i], leaf_tar,
                                                                                       idx_fid_is)),
                                      padding=0, normalize=True)
                    vutils.save_image(img_l1.data, os.path.join(output_directory2,
                                      'level1/{:03d}_{:02d}To{:02d}_{:03d}.jpg'.format(count, h1_labels[i], inter_tar,
                                                                                       idx_fid_is)),
                                      padding=0, normalize=True)
                    vutils.save_image(img_l0.data, os.path.join(output_directory2,
                                      'level0/{:03d}_{:02d}To{:02d}_{:03d}.jpg'.format(count, h2_labels[i], root_tar,
                                                                                       idx_fid_is)),
                                      padding=0, normalize=True)


    sys.exit('Finish test')

