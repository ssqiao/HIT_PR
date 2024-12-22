from __future__ import print_function
from utils import get_config
from trainer import HIT_Trainer
import torch.backends.cudnn as cudnn
import argparse
import matplotlib.pyplot as plt
from confidence_ellipse import confidence_ellipse
# center xy, width left 2 right, height bottom to top, alpha anti-clockwise angle
import numpy as np
import umap
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import torch
import os

#   https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',
          '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9']
# ImageNet qss
classes = ['Egyptian', 'Persian', 'Siamese', 'Tabby', 'Corgi', 'GermanShepherd', 'Husky', 'Samoyed', 'Cougar',
           'Leopard', 'Lion', 'Tiger', 'House Cats', 'Dogs', 'Big Cats', 'Animals']
show_c = [1, 7, 12, 13, 15]
# ShapeNet
# classes = ['Loveseat', 'Club chair', 'Leather', 'Worktable', 'Billiard', 'Tennis', 'Sofa', 'Table', 'Furniture']
# show_c = [1, 5, 6, 7, 8]

color_index = [0, 8, 7, 1, 4]
maker = ['.',  '*', '+']
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/imagenet.yaml', help='Path to the config file.')
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=1, help="random seed")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
np.random.seed(opts.seed)
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
# Setup model and data loader
trainer = HIT_Trainer(config)

root_num = trainer.gen.distributions.root_num
inter_num = trainer.gen.distributions.intermediate_num
leaf_num = trainer.gen.distributions.leaves_num
total_num = root_num + inter_num + leaf_num


model_name = config['model_name']
output_directory = os.path.join("./results", model_name)

if not os.path.exists(output_directory):
    print("Creating directory: {}".format(output_directory))
    os.makedirs(output_directory)


state_dict = torch.load(opts.checkpoint)
trainer.gen.load_state_dict(state_dict['gen'])


trainer.cuda()
trainer.eval()
distributions_tree = trainer.gen.distributions

num_per_dist = 1000

with torch.no_grad():
    samples = distributions_tree.forward(sample_size=num_per_dist)
    samples = samples.cpu().numpy()
# tSNE vis
fea = samples
X_tsne = umap.UMAP(n_neighbors=5,
                      min_dist=0.1,
                      metric='correlation').fit_transform(fea)

fig, ax = plt.subplots(figsize=(12, 6))
for count, c_id in enumerate(show_c):
    tmp_tsne = X_tsne[num_per_dist*c_id:num_per_dist*(c_id+1)]
    plt.scatter(x=tmp_tsne[:, 0], y=tmp_tsne[:, 1], alpha=1, color=colors[color_index[count]], marker=maker[count//2])
    confidence_ellipse(x=tmp_tsne[:, 0], y=tmp_tsne[:, 1], ax=ax, label=classes[c_id], alpha=1, edgecolor=colors[color_index[count]],
                       n_std=2., linewidth=2)

ax.axvline(c='grey', lw=1)
ax.axhline(c='grey', lw=1)
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(os.path.join(output_directory, 'umap_distributions.pdf'))
plt.show()
plt.close()
