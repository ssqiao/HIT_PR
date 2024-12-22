import os
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as Lr_Sheduler
import matplotlib
from data import ImageLabelFilelist
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

'''
Train standard AlexNet from scratch with 128*128 resolution and middle level category.
Data provide from list file 
Net define
Train loop and print info.
'''


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


batch_size = 128
# qss
input_image_size = 256
crop_size = 224
gpus = '0'
lst_gpu = list(gpus.split(','))
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

# qss
save_dir = '/home/ouc/data1/qiaoshishi/python_codes/MDIT/AlexNet_animal_face_super_cls_cat_dog_bigcat_bear_wolf_fox/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# data prepare
root_dir = '/home/ouc/data1/qiaoshishi/datasets/imageNet_translation/'
# qss
train_file_list = 'animals_list_train_closeset.txt'
test_file_list = 'animals_list_test_closeset.txt'
# root_dir = root_dir + 'cropped_extend'
tree = [6, 4, 4, 4, 4, 4, 4, 1, 6]
filter_label = [146, 145, 106, 144, 140, 69, 83, 88, 113, 147, 111, 112, 114, 115, 116, 117, 97, 96, 98, 95, 101, 103, 102, 104]
num_classes = 6
train_transforms = transforms.Compose([
    transforms.Resize((input_image_size, input_image_size)),
    transforms.RandomCrop(crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # qss
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((input_image_size, input_image_size)),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    # qss
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
])

# qss
train_dataset = ImageLabelFilelist(root_dir, train_file_list, filter_label, tree, transform=train_transforms,
                                   shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                           num_workers=4)
test_dataset = ImageLabelFilelist(root_dir, test_file_list, filter_label, tree, transform=test_transforms,
                                   shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                           num_workers=4)




# qss, net define
class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # qss pretrained
        self.alex = torchvision.models.alexnet(pretrained=True)
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
criterion = torch.nn.CrossEntropyLoss().cuda()
if len(lst_gpu) > 1:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()


# qss
step = 0
start_epoch = 0
num_epoch = 30
resume_epoch = start_epoch

lr_nn = 0.001
# params_list = [{'params': net.feature_layers.parameters(), 'lr': 0.1*lr_nn},
#                    {'params': net.fc.parameters()}]
optimizer_nn = optim.SGD(net.parameters(), lr=lr_nn, momentum=0.9, weight_decay=5e-4)
# qss
lr_sheduler_nn = Lr_Sheduler.MultiStepLR(optimizer_nn, milestones=[5, 10, 15, 20, 25], gamma=0.5)
# def adjust_learning_rate(optimizer, epoch):
#     lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
#     optimizer.param_groups[0]['lr'] =0.1*lr
#     optimizer.param_groups[1]['lr'] = lr
hist_loss = []
hist_acc = []
hist_test_acc = []


# def check_cuda(arg):
    # arg = arg.cuda(async=True)
    # return arg


def train(epoch, step):
    net.train()
    start_time = time.time()
    for it, (images, h0_labels, h1_labels, h2_labels, h0_local_labels, h1_local_labels) in enumerate(train_loader):
        # Main training code, debug
        inputs = images.cuda().detach()
        targets = h1_labels.cuda().detach()
        if len(lst_gpu) > 1:
            inputs = inputs.cuda()
            targets = targets.cuda()
            # fnames = check_cuda(fnames)
        # inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        logits = net(inputs)
        loss = criterion(logits, targets)

        optimizer_nn.zero_grad()
        loss.backward()
        optimizer_nn.step()

        # train_loss += loss.data[0]
        _, predicted = torch.max(logits.data, 1)
        accuracy = (targets.data == predicted).cpu().float().mean()

        if step % 10 == 0:
            np_loss = loss.item()
            np_acc = accuracy
            hist_loss.append(np_loss)
            hist_acc.append(np_acc)

            # net.eval()
            with torch.no_grad():
                total_num = 0
                count = 0
                for _, (test_images, _, test_h1_labels, _, _, _) in enumerate(test_loader):
                    inputs = test_images.cuda().detach()
                    targets = test_h1_labels.cuda().detach()
                    total_num += inputs.size(0)
                    if len(lst_gpu) > 1:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                        # fnames = check_cuda(fnames)
                    predictions = net(inputs)

                    _, predicted = torch.max(predictions.data, 1)
                    count += (targets.data == predicted).cpu().float().sum()

                hist_test_acc.append(count / total_num)
            print(
                    '[epoch %d step %d] loss %f acc %f test_acc %f time %f' % (
            epoch, step, np_loss, np_acc, count / total_num,  time.time() - start_time))
            start_time = time.time()

            plt.subplot(131)
            plt.plot(list(range(len(hist_loss))), hist_loss)
            plt.title('Loss')
            plt.grid(True)

            plt.subplot(132)
            plt.plot(list(range(len(hist_acc))), hist_acc)
            plt.title('Accuracy')
            plt.grid(True)

            plt.subplot(133)
            plt.plot(list(range(len(hist_test_acc))), hist_test_acc)
            plt.title('Test Accuracy')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'loss_acc_3.jpg'))
            plt.close()

        step += 1
    return step


if step > 0:
    checkpoint = torch.load(os.path.join(save_dir, 'step_' + str(step) + '.model'))
    resume_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net_state_dict'])
    print('resume')

start_epoch = 0
for epoch in range(start_epoch, start_epoch + num_epoch):

    lr_sheduler_nn.step()
    print('\nEpoch: %d of %d' % (epoch, num_epoch))
    print('learning rate')
    for param_group in optimizer_nn.param_groups:
        print(param_group['lr'])
    if epoch < resume_epoch:
        continue

    step = train(epoch, step)
    save_checkpoint({
        'epoch': epoch,
        'step': step,
        'net_state_dict': net.state_dict(),
        'hist_loss': hist_loss,
        'hist_acc': hist_acc,
    }, filename=os.path.join(save_dir, 'step_' + str(step) + '.model'))
