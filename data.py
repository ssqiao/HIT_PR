import torch.utils.data as data
import os.path
from PIL import Image
import numpy as np


# open an image & convert to RGB format
def default_loader(path):
    return Image.open(path).convert('RGB')


# read txt list path+label
def default_flist_reader1(flist):
    """
    flist format: impath label\n impath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            # impath = line.strip()  # delete space char at start and end locations
            # imlist.append(impath)
            imlist.append(line)

    return imlist


# filter and leave the desired classes
def default_flist_reader(data_dir, file_name, filter_lab):
    file_path = os.path.join(data_dir, file_name)
    print('Getting file list!')
    with open(file_path) as fid:
        content = fid.read()
        contentList = content.split('\n')

    fileList = []
    label = []
    index = []

    for term in contentList:
        tmp = term.split()
        fileList.append(tmp[0])
        label.append(int(tmp[1]))

    label = np.array(label)
    if filter_lab is None:
        index = range(len(fileList))
    else:
        for lab in filter_lab:
            tmp_list = np.argwhere(label == lab).tolist()
            index += tmp_list

    fileList = np.array(fileList)
    fileList = fileList[index]
    fileList = np.reshape(fileList, [-1])
    label = label[index]
    fileList = [data_dir + x for x in fileList]

    return fileList, label


# get the super-category labels
def get_super_category(basic_label_list, tree, filter_lab=None):
    output = []
    num_super_category = tree[0]

    for lab in basic_label_list:
        if filter_lab is not None:
            lab = filter_lab.index(lab)
        if lab < tree[1]:
            lab = 0
            output.append(int(lab))
            continue
        for i in range(2, num_super_category+1):
            tmp = sum(tree[1:i+1])
            if lab < tmp:
                lab = i - 1
                output.append(int(lab))
                break
    return np.array(output)


# scale the filtered classes start with label 0 in ascending order
def scale_label(file_list, label, filter_lab, shuffle=False):
    for index1, value in enumerate(label[0]):
        if filter_lab is None:
            idx = value
        else:
            idx = filter_lab.index(value)

        file_list[index1] = file_list[index1] + ' ' + str(idx)
        for i in range(1, len(label)):
            file_list[index1] = file_list[index1] + ' ' + str(label[i][index1])

    tmp = np.array(file_list)

    if shuffle:
        # index = range(len(file_list))
        np.random.shuffle(tmp)
        # tmp = tmp[index]

    return tmp.tolist()

# scale the local classes (brothers) of a particular hierarchy to start with label 0 in ascending order
def scale_local_label(label_list, tree, filter_lab = None):
    tmp_list = np.squeeze(np.array(label_list))
    if filter_lab is not None:
        for idx1, val in enumerate(tmp_list):
            tmp_list[idx1] = filter_lab.index(val)

    scale_factor = 0
    for idx in range(tree[0]):
        tmp_index = np.argwhere((scale_factor <= tmp_list) * (tmp_list < scale_factor+tree[idx+1]))
        tmp_list[tmp_index] = tmp_list[tmp_index] - scale_factor
        scale_factor += tree[idx+1]

    return np.array(tmp_list)

# inherit from abstract class Dataset and override the getitem and len func.
# root path, txt path (contain image path & labels), txt list reader, image loader
class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, filter_label, tree, transform=None, shuffle=False,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.tree = tree
        self.filter_label = filter_label
        self.path_list, self.label_list = flist_reader(self.root, flist, self.filter_label)
        local_label_list = scale_local_label(self.label_list, tree, filter_label)
        h1_label_list = get_super_category(self.label_list, self.tree, self.filter_label)
        h1_local_label_list = scale_local_label(h1_label_list, tree[tree[0] + 1:])
        h2_label_list = get_super_category(h1_label_list, self.tree[self.tree[0]+1:])
        self.imgs = scale_label(self.path_list, [self.label_list, h1_label_list, h2_label_list, local_label_list,
                                h1_local_label_list], self.filter_label, shuffle)
        self.transform = transform
        self.loader = loader
        # self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        term = self.imgs[index]
        tmp = term.split(' ')
        img = self.loader(tmp[0])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), int(tmp[5])

    def __len__(self):
        return len(self.label_list)


class ImageLabelList(data.Dataset):
    def __init__(self, root, flist, transform=None, flist_reader=default_flist_reader1, loader=default_loader):
        self.root = root
        self.imgs = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        # self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        term = self.imgs[index]
        tmp = term.split(' ')
        img = self.loader(tmp[0])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(tmp[1])

    def __len__(self):
        return len(self.imgs)


