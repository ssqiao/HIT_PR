from __future__ import print_function
from data import ImageLabelList
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import torch
import os, sys
"""
Test the acc of generated images in middle level category
"""

batch_size = 128
# qss
input_image_size = 256
crop_size = 224
gpus = '2'
lst_gpu = list(gpus.split(','))
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

# qss
root_dir = '/home/ouc/data1/qiaoshishi/python_codes/MDIT/results/imagenet_maha1_cls_cycle_entropy_distw0.1/user_study/level1/'
test_file_list = 'img_label_list.txt'
# root_dir = root_dir + 'cropped_extend'
num_classes = 3  # 4 for shapenet

test_transforms = transforms.Compose([
    transforms.Resize((input_image_size, input_image_size)),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    # qss
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
])

# qss
test_dataset = ImageLabelList(root_dir, test_file_list, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                           num_workers=4)

# qss, net define
class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # qss pretrained
        self.alex = torchvision.models.alexnet(pretrained=False)
        # qss
        self.alex.classifier = torch.nn.Sequential(*list(self.alex.classifier.children())[:6])
        self.fc = torch.nn.Linear(4096, num_classes, bias=False)
        self.feature_layers = torch.nn.Sequential(self.alex.features, self.alex.classifier)

    def forward(self, x):
        x = self.alex.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.alex.classifier(x)
        output = self.fc(x)
        return output

net = CNN(num_classes=num_classes)
if len(lst_gpu) > 1:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

# qss
checkpoint = 'AlexNet_imagenet_super_cls/step_810.model'
# try:
state_dict = torch.load(checkpoint)
net.load_state_dict(state_dict['net_state_dict'])
print('resume')
# except:
#     state_dict = pytorch03_to_pytorch04(torch.load(checkpoint), net)
#     net.load_state_dict(state_dict['net_state_dict'])


net.eval()
# Start testing
test_count = 0
test_sample_num = test_loader.dataset.__len__()
correct_test = 0.

with torch.no_grad():
    for _, (test_data) in enumerate(test_loader):
        test_images, test_labels = test_data[0].cuda(),  test_data[1].cuda()

        test_images = Variable(test_images.cuda())
        predictions = net(test_images)

        _, predicted = torch.max(predictions.data, 1)
        test_count += (test_labels.data == predicted).float().sum()

    acc_test_h1 = test_count/test_sample_num*100
    print('acc_test_h1:{}\n '.format(acc_test_h1))

    sys.exit('Finish test')

