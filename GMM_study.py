from __future__ import print_function
from utils import get_config
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
from sklearn.mixture import GaussianMixture

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/cadcars.yaml', help='Path to the config file.')
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--res_path', type=str, default='.', help="path for resnet34 model weight")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
np.random.seed(opts.seed)
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
config['res_model_path'] = opts.res_path
# Setup model and data loader
trainer = HIT_Trainer(config)

root_num = trainer.gen.distributions.root_num
root_tar = np.random.randint(low=0, high=root_num)
inter_num = trainer.gen.distributions.intermediate_num
inter_tar = np.random.randint(low=0, high=inter_num)
leaf_num = trainer.gen.distributions.leaves_num
leaf_tar = np.random.randint(low=0, high=leaf_num)
total_num = root_num + inter_num + leaf_num

_, test_loader = utl.get_all_data_loaders(config)

model_name = config['model_name']
output_directory1 = os.path.join("./results", model_name, 'GMM_study', 'GMM_fit')
output_directory2 = os.path.join("./results", model_name, 'GMM_study', 'Gaussian_learned')
output_directory3 = os.path.join("./results", model_name, 'GMM_study', 'Gaussian_fit')

if not os.path.exists(output_directory1):
    print("Creating directory: {}".format(output_directory1))
    os.makedirs(output_directory1)

if not os.path.exists(output_directory2):
    print("Creating directory: {}".format(output_directory2))
    os.makedirs(output_directory2)

if not os.path.exists(output_directory3):
    print("Creating directory: {}".format(output_directory3))
    os.makedirs(output_directory3)


state_dict = torch.load(opts.checkpoint)
trainer.gen.load_state_dict(state_dict['gen'])



trainer.cuda()
trainer.eval()
encode = trainer.gen.encode
decode = trainer.gen.decode
distributions_tree = trainer.gen.distributions

# Start testing
test_img_num = 100
num_per_dist = 5000  # qss
fid_is_style_num = 100  # qss
max_test_num = 500
count = 0
sample_num = test_loader.dataset.__len__()


with torch.no_grad():
    style_sampled_all = distributions_tree.forward(sample_size=num_per_dist)
    tar_level_children_styles = style_sampled_all[leaf_num * num_per_dist:(leaf_num + inter_num) * num_per_dist].cpu().numpy()

    # parent GMM fit and sampling
    gmm = GaussianMixture(n_components=inter_num, covariance_type='diag', random_state=opts.seed)
    gmm.fit(tar_level_children_styles)
    print(gmm.converged_)

    # parent learned Gaussian sampling, qss
    # style_sampled_gaussian_learned = style_sampled_all[(leaf_num + inter_num) * num_per_dist:
    #                                                    (leaf_num + inter_num) * num_per_dist + fid_is_style_num]

    # parent single Gaussian fit and sampling
    mean = np.mean(tar_level_children_styles, 0)
    cov = np.cov(tar_level_children_styles, rowvar=False)

    for _, (images, h0_labels, h1_labels, h2_labels, _, _) in enumerate(test_loader):
        style_sampled_gmm = gmm.sample(fid_is_style_num)[0].astype(np.float32)
        style_sampled_gmm = torch.from_numpy(style_sampled_gmm).cuda()

        style_sampled_all = distributions_tree.forward(sample_size=fid_is_style_num)
        style_sampled_gaussian_learned = style_sampled_all[(leaf_num + inter_num) * fid_is_style_num:
                                                           (leaf_num + inter_num) * fid_is_style_num + fid_is_style_num]

        style_sampled_gaussian_fit = np.random.multivariate_normal(mean, cov, fid_is_style_num).astype(np.float32)
        style_sampled_gaussian_fit = torch.from_numpy(style_sampled_gaussian_fit).cuda()

        images = Variable(images.cuda())
        content, _ = encode(images)
        batch_size = images.size(0)
        for i in range(batch_size):
            if count >= max_test_num or count >= sample_num:
                sys.exit('Finish test')
            count = count + 1
            # for debug
            # vutils.save_image(images[i].data, os.path.join(input_directory, '{:03d}_{:02d}_{:02d}_{:02d}.jpg'.
            #                                                format(count, h0_labels[i], h1_labels[i], h2_labels[i])),
            #                   padding=0, normalize=True)
            c1 = content[i].unsqueeze(0)

            # for quantitative test
            if count <= test_img_num:
                c_fid_is = c1.repeat(fid_is_style_num, 1, 1, 1)

                x_fid_is_l2 = decode(c_fid_is, style_sampled_gmm)
                x_fid_is_l1 = decode(c_fid_is, style_sampled_gaussian_learned)
                x_fid_is_l0 = decode(c_fid_is, style_sampled_gaussian_fit)
                for idx_fid_is, (img_l2, img_l1, img_l0) in enumerate(zip(x_fid_is_l2, x_fid_is_l1, x_fid_is_l0)):
                    vutils.save_image(img_l2.data, os.path.join(output_directory1,
                                      '{:03d}_{:03d}.jpg'.format(count, idx_fid_is)),
                                      padding=0, normalize=True)
                    vutils.save_image(img_l1.data, os.path.join(output_directory2,
                                      '{:03d}_{:03d}.jpg'.format(count, idx_fid_is)),
                                      padding=0, normalize=True)
                    vutils.save_image(img_l0.data, os.path.join(output_directory3,
                                      '{:03d}_{:03d}.jpg'.format(count, idx_fid_is)),
                                      padding=0, normalize=True)

    sys.exit('Finish test')

