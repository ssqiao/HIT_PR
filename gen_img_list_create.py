import os
import glob

"""
Obtain the file list file with img_path + label format for translated images in a foloder 
"""

# qss
img_folder = '/home/ouc/data1/qiaoshishi/python_codes/MDIT/results/imagenet_maha1_cls_cycle_entropy_distw0.1/user_study/level1/'
img_name_list = glob.glob(img_folder + '*.jpg')

with open(os.path.join(img_folder,'img_label_list.txt'), 'w+') as f:
    for line in img_name_list:
        # parse and write to f
        idx = line.index('To')
        tmp = line[idx+2:idx+4]  # qss
        lab = int(tmp) - 1
        # lab = 2
        f.write(line+' '+str(lab)+'\n')
f.close()