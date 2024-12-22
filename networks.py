from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass

# Generator- AdaINGen, call Decoder, Encoder, sequential blocks
# Multi-scale discriminator- MsImageDis, call basic blocks

# Encoder- StyleEncoder, ContentEncoder, call sequential and basic blocks
# NestedDistributions, call Node
# Decoder, call sequential and basic blocks

# Sequential- ResBlocks, MLP modules, call basic blocks
# Basic blocks- ResBlock, Conv2dBlock, LinearBlock, AdaptiveInstanceNorm2d, LayerNorm


##################################################################################
# Discriminator
##################################################################################

# qss auxiliary classifier
class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        if 'multi_adv' in params.keys():
            self.multi_adv = params['multi_adv']
        else:
            self.multi_adv = False
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.tree = params['tree']
        self.h2_classes = self.tree[self.tree[0] + 1]
        self.h1_classes = self.tree[0]
        self.h0_classes = sum(self.tree[1:self.tree[0] + 1])
        self.cnns = nn.ModuleList()
        self.aux_h0 = nn.ModuleList()
        self.aux_h1 = nn.ModuleList()
        if self.h2_classes > 1:
            self.aux_h2 = nn.ModuleList()
        self.dis = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())
            self.aux_h0.append(nn.Conv2d(self.dim * (2 ** (self.n_layer-1)), self.h0_classes, 1, 1, 0))
            self.aux_h1.append(nn.Conv2d(self.dim * (2 ** (self.n_layer-1)), self.h1_classes, 1, 1, 0))
            self.dis.append(nn.Conv2d(self.dim * (2 ** (self.n_layer-1)), 1, 1, 1, 0))
            if self.h2_classes > 1:
                self.aux_h2.append(nn.Conv2d(self.dim*(2**(self.n_layer-1)), self.h2_classes, 1, 1, 0))
        self.mlp = []
        # qss style dim 8,16 when use style adv loss and change the style dim
        self.mlp += [LinearBlock(8, 64, norm=self.norm, activation=self.activ)]
        self.mlp += [LinearBlock(64, 64, norm=self.norm, activation=self.activ)]
        self.mlp = nn.Sequential(*self.mlp)
        self.dis_style = LinearBlock(64, 1, norm='none', activation='none')
        self.aux_style_h0 = LinearBlock(64, self.h0_classes, norm='none', activation='none')

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        if self.norm == 'sn':
            cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        else:
            cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ,
                                  pad_type=self.pad_type)]

        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs_dis, outputs_aux_h0, outputs_aux_h1, outputs_aux_h2 = [], [], [], []
        for it, (cnn, cls_h0, cls_h1, dis) in enumerate(zip(self.cnns, self.aux_h0, self.aux_h1, self.dis)):
            tmp = cnn(x)
            tmp_h0 = cls_h0(tmp)
            tmp_h0 = self.global_avg_pool(tmp_h0).squeeze()
            tmp_h1 = cls_h1(tmp)
            tmp_h1 = self.global_avg_pool(tmp_h1).squeeze()
            tmp_dis = dis(tmp)
            outputs_aux_h0.append(tmp_h0)
            outputs_aux_h1.append(tmp_h1)
            outputs_dis.append(tmp_dis)
            if self.h2_classes > 1:
                tmp_h2 = self.aux_h2[it](tmp)
                tmp_h2 = self.global_avg_pool(tmp_h2).squeeze()
                outputs_aux_h2.append(tmp_h2)
            x = self.downsample(x)
        return outputs_dis, outputs_aux_h0, outputs_aux_h1, outputs_aux_h2

    def calc_dis_loss(self, input_fake, input_real, h0_labels=None, h0_local_labels=None, h1_labels=None,
                      h1_local_labels=None, h2_labels=None, local_cls=False, h0_labels_fake=None,
                      h0_local_labels_fake=None, h1_labels_fake=None, h1_local_labels_fake=None, h2_labels_fake=None):
        # calculate the loss to train D
        outs0, outs0_h0, outs0_h1, outs0_h2 = self.forward(input_fake)
        outs1, outs1_h0, outs1_h1, outs1_h2 = self.forward(input_real)
        loss_adv, loss_h0, loss_h1, loss_h2 = 0., 0., 0., 0.

        # alpha = torch.rand(input_real.size(0), 1)
        # alpha = alpha.expand(input_real.size(0), input_real.nelement() / input_real.size(0)).contiguous().\
        #     view(input_real.size())
        # alpha = alpha.cuda()
        #
        # interpolates = alpha * input_real + ((1 - alpha) * input_fake)
        # interpolates = interpolates.cuda()
        # interpolates = Variable(interpolates, requires_grad=True)
        # disc_interpolates, _, _, _ = self.forward(interpolates)

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.multi_adv:  # qss
                continue
            if self.gan_type == 'lsgan':
                loss_adv += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':  # original gan ?
                # all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                # all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                # loss_adv += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                #                        F.binary_cross_entropy(F.sigmoid(out1), all1))
                loss_adv += F.softplus(-out1).mean() + F.softplus(out0).mean()
            elif self.gan_type == 'wgan':
                loss_adv += F.relu(1.0 - out1).mean() + F.relu(1.0 + out0).mean()
            #     gradients = torch.autograd.grad(outputs=out_inter, inputs=interpolates,
            #                                     grad_outputs=torch.ones(out_inter.size()).cuda(),
            #                                     create_graph=True, retain_graph=True, only_inputs=True)[0]
            #     gradients = gradients.view(gradients.size(0), -1)
            #     loss_adv += ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.0
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        if h0_local_labels is not None:
            if not self.multi_adv:
                loss_h0 = self.calc_auxiliary_loss_h0(outs1_h0, h0_local_labels.cuda(), h1_labels.cuda(),
                                                      h0_labels.cuda(), local_cls)
            else:
                if h0_labels_fake is not None:
                    loss_h0 = self.calc_multi_dis_adv_loss(outs1_h0, outs0_h0, h0_labels, h0_labels_fake)

        if h1_local_labels is not None:
            if not self.multi_adv:
                loss_h1 = self.calc_auxiliary_loss_h1(outs1_h1, h1_local_labels.cuda(), h2_labels.cuda(),
                                                      h1_labels.cuda(), local_cls)
            else:
                if h1_labels_fake is not None:
                    loss_h1 = self.calc_multi_dis_adv_loss(outs1_h1, outs0_h1, h1_labels, h1_labels_fake)

        if (self.h2_classes > 1 or self.multi_adv) and h2_labels is not None:
            if not self.multi_adv:
                loss_h2 = self.calc_auxiliary_loss_h2(outs1_h2, h2_labels.cuda())
            else:
                if h2_labels_fake is not None:
                    loss_h2 = self.calc_multi_dis_adv_loss(outs1_h2, outs0_h2, h2_labels, h2_labels_fake)

        return loss_adv, loss_h0, loss_h1, loss_h2

    def calc_gen_loss(self, input_fake, h0_labels=None, h0_local_labels=None, h1_labels=None, h1_local_labels=None,
                      h2_labels=None, local_cls=False, per_num=1, entropy_h0=False, entropy_h1=False):
        # calculate the loss to train G
        outs0, outs_h0, outs_h1, outs_h2 = self.forward(input_fake)
        loss_adv, loss_aux_h0, loss_aux_h1, loss_aux_h2 = 0., 0., 0., 0.
        loss_instance_entropy, loss_avg_entropy = 0., 0.
        for it, (out0) in enumerate(outs0):
            if self.multi_adv:  # qss
                continue
            if self.gan_type == 'lsgan':
                loss_adv += torch.mean((out0 - 1)**2)  # LSGAN
            elif self.gan_type == 'nsgan':
                # all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                # loss_adv += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
                loss_adv += F.softplus(-out0).mean()
            elif self.gan_type == 'wgan':
                loss_adv += -torch.mean(out0)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        if h0_local_labels is not None:
            if not self.multi_adv:
                loss_aux_h0 = self.calc_auxiliary_loss_h0(outs_h0, h0_local_labels.cuda(), h1_labels.cuda(),
                                                          h0_labels.cuda(), local_cls)
            else:
                loss_aux_h0 = self.calc_multi_gen_adv_loss(outs_h0, h0_labels)
        if h1_local_labels is not None:
            if not self.multi_adv:
                loss_aux_h1 = self.calc_auxiliary_loss_h1(outs_h1, h1_local_labels.cuda(), h2_labels.cuda(),
                                                          h1_labels.cuda(), local_cls)
            else:
                loss_aux_h1 = self.calc_multi_gen_adv_loss(outs_h1, h1_labels)
        if h2_labels is not None and (self.h2_classes >1 or self.multi_adv):
            if not self.multi_adv:
                loss_aux_h2 = self.calc_auxiliary_loss_h2(outs_h2, h2_labels.cuda())
            else:
                loss_aux_h2 = self.calc_multi_gen_adv_loss(outs_h2, h2_labels)

        if per_num > 1 and entropy_h0 and not entropy_h1:
            loss_instance_entropy, loss_avg_entropy = self.compute_entropy_loss(outs_h0, h1_labels, per_num=per_num)
        elif per_num > 1 and entropy_h0 and entropy_h1:
            loss_instance_entropy, loss_avg_entropy = self.compute_entropy_loss(outs_h0=outs_h0, h1_labels=None,
                                                                                outs_h1=outs_h1, h2_labels=h2_labels,
                                                                                per_num=per_num)
        else:
            pass

        return loss_adv, loss_aux_h0, loss_aux_h1, loss_aux_h2, loss_instance_entropy, loss_avg_entropy

    def calc_dis_style_loss(self, style_real, style_fake, h0_labels):
        loss_adv, loss_aux = 0., 0.
        inter_fea_real = self.mlp(style_real)
        inter_fea_fake = self.mlp(style_fake.view(style_fake.size(0), -1))
        out_real = self.dis_style(inter_fea_real)
        out_fake = self.dis_style(inter_fea_fake)
        out_aux = self.aux_style_h0(inter_fea_real)

        if self.gan_type == 'lsgan':
            loss_adv += torch.mean((out_fake - 0) ** 2) + torch.mean((out_real - 1) ** 2)
        elif self.gan_type == 'nsgan':  # original gan ?
            all0 = Variable(torch.zeros_like(out_fake.data).cuda(), requires_grad=False)
            all1 = Variable(torch.ones_like(out_real.data).cuda(), requires_grad=False)
            loss_adv += torch.mean(F.binary_cross_entropy(F.sigmoid(out_fake), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out_real), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss_aux += torch.mean(F.cross_entropy(out_aux, h0_labels))

        return loss_adv, loss_aux

    def calc_gen_style_loss(self, style_fake, h0_labels):
        loss_adv, loss_aux = 0., 0.
        inter_fea = self.mlp(style_fake.view(style_fake.size(0), -1))
        out = self.dis_style(inter_fea)
        out_aux = self.aux_style_h0(inter_fea)

        if self.gan_type == 'lsgan':
            loss_adv += torch.mean((out - 1) ** 2)  # LSGAN
        elif self.gan_type == 'nsgan':
            all1 = Variable(torch.ones_like(out.data).cuda(), requires_grad=False)
            loss_adv += torch.mean(F.binary_cross_entropy(F.sigmoid(out), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss_aux += torch.mean(F.cross_entropy(out_aux, h0_labels))

        return loss_adv, loss_aux

    def calc_multi_dis_adv_loss(self, outs_real, outs_fake, real_labels, fake_labels):
        loss = 0.
        idx = torch.LongTensor(range(real_labels.size(0))).cuda()
        for it, (out_real, out_fake) in enumerate(zip(outs_real, outs_fake)):
            # channel_one_hot_real = self.one_hot_labels(real_labels, out_real)
            # channel_one_hot_fake = torch.zeros_like(out_fake)
            # loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out_fake), channel_one_hot_fake) +
            #                    F.binary_cross_entropy(F.sigmoid(out_real), channel_one_hot_real))
            out_real_select = out_real[idx, real_labels]
            out_fake_select = out_fake[idx, fake_labels]
            target_ones = torch.full_like(out_real_select, fill_value=1)
            target_zeros = torch.full_like(out_real_select, fill_value=0)
            if self.gan_type == 'nsgan':  # qss
                # loss += torch.mean(F.binary_cross_entropy_with_logits(out_real_select, target_ones) +
                #                    F.binary_cross_entropy_with_logits(out_fake_select, target_zeros))
                loss += F.softplus(-out_real_select).mean() + F.softplus(out_fake_select).mean()
            elif self.gan_type == 'lsgan':
                loss += torch.mean((out_fake_select - 0)**2) + torch.mean((out_real_select - 1)**2)
            elif self.gan_type == 'wgan-gp':
                loss += F.relu(1.0 - out_real_select).mean() + F.relu(1.0 + out_fake_select).mean()

        return loss

    def calc_multi_gen_adv_loss(self, outs, labels):
        loss = 0.
        for it, (out) in enumerate(outs):
            out_select = out[range(labels.size(0)), labels]
            # channel_one_hot = self.one_hot_labels(labels, out)
            all1 = torch.full_like(out_select, fill_value=1)
            # loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out), channel_one_hot))
            if self.gan_type == 'nsgan':  # qss
                # loss += torch.mean(F.binary_cross_entropy_with_logits(out_select, all1))
                loss += F.softplus(-out_select).mean()
            elif self.gan_type == 'lsgan':
                loss += torch.mean((out_select - 1)**2)
            elif self.gan_type == 'wgan-gp':
                loss += -out_select.mean()

        return loss

    def compute_local_cls_loss(self, logits, local_idx, class_num, local_labels, parent_labels):
        cls_local_loss = 0.
        local_logits = torch.split(logits, local_idx, 1)
        for lab in range(class_num):
            tmp = torch.eq(parent_labels, lab)
            flag = torch.sum(tmp)
            if flag > 0:
                idx = torch.nonzero(tmp)
                select_local_labels = local_labels[idx].squeeze(1)
                select_local_logits = local_logits[lab][idx,:].squeeze(1)
                cls_local_loss += torch.sum(F.cross_entropy(select_local_logits, select_local_labels))

        return cls_local_loss/logits.size(0)

    def negative_entropy_prediction(self, logits, tree=None, softmax=True, num=1):
        eps = 1e-9
        # entropy of all categories in the level
        if tree is None:
            first_child = F.softmax(logits) if softmax else logits
            if num > 1:
                shape = first_child.shape
                first_child = first_child.reshape(-1,  num, shape[1])
                first_child = torch.mean(first_child, 1)
            log_first_child = torch.log(first_child + eps)
            element_entropy = first_child * log_first_child
            first_child_entropy = torch.sum(element_entropy, 1).unsqueeze(1)
            entropy = first_child_entropy
            return entropy

        # entropy of local brothers for each super category
        num_supercategory = tree[0]

        splited_logits = torch.split(logits, tree[1:num_supercategory + 1], dim=-1)

        first_child = F.softmax(splited_logits[0]) if softmax else splited_logits[0]
        if num > 1:
            first_child = first_child.reshape(-1, num, tree[1])
            first_child = torch.mean(first_child, 1)
        log_first_child = torch.log(first_child + eps)
        element_entropy = first_child * log_first_child
        first_child_entropy = torch.sum(element_entropy, 1).unsqueeze(1)
        entropy = first_child_entropy

        for i in range(1, num_supercategory):
            tmp_child = F.softmax(splited_logits[i]) if softmax else splited_logits[i]
            if num > 1:
                tmp_child = tmp_child.reshape(-1, num, tree[i+1])
                tmp_child = torch.mean(tmp_child, 1)
            log_tmp_child = torch.log(tmp_child + eps)
            tmp_element_entropy = tmp_child * log_tmp_child
            tmp_child_entropy = torch.sum(tmp_element_entropy, 1).unsqueeze(1)
            entropy = torch.cat((entropy, tmp_child_entropy), 1)

        return entropy

    def compute_entropy_loss(self, outs_h0, h1_labels=None, outs_h1=None, h2_labels=None, per_num=5):
        instance_entropy_loss, avg_entropy_loss = 0., 0.
        # currently design for global cls in each level
        # level = 1
        if h1_labels is not None:
            h1_one_hot = torch.Tensor(h1_labels.size(0), self.h1_classes).cuda().detach()
            h1_one_hot.zero_()
            h1_one_hot.scatter_(1, h1_labels.unsqueeze(1), 1)
            h1_one_hot_avg = h1_one_hot.reshape(-1, per_num, self.h1_classes)
            h1_one_hot_avg = torch.mean(h1_one_hot_avg, 1)

            for it, (out_h0) in enumerate(outs_h0):
                out_h0 = F.softmax(out_h0)
                global_entropy = - torch.mean(self.negative_entropy_prediction(out_h0, softmax=False))
                # local_entropy = - torch.mean(torch.sum(self.negative_entropy_prediction(out_h0, self.tree,
                #                                                                         softmax=False)*h1_one_hot, 1))

                out_h0 = out_h0.reshape(-1, per_num, self.h0_classes)
                out_h0 = torch.mean(out_h0, 1)
                local_entropy_avg = torch.mean(torch.sum(self.negative_entropy_prediction(out_h0, self.tree, False)*
                                                         h1_one_hot_avg, 1))
                instance_entropy_loss += global_entropy
                avg_entropy_loss += local_entropy_avg
        # level = 0
        elif h2_labels is not None:
            if self.h2_classes>1:
                h2_one_hot = torch.Tensor(h2_labels.size(0), self.h2_classes).cuda().detach()
                h2_one_hot.zero_()
                h2_one_hot.scatter_(1, h2_labels.unsqueeze(1), 1)
                h2_one_hot_avg = h2_one_hot.reshape(-1,per_num, self.h2_classes)
                h2_one_hot_avg = torch.mean(h2_one_hot_avg, 1)
                tmp_tree = list([])
                tmp_tree.append(self.h2_classes)
                idx = 1
                for i in range(self.h2_classes):
                    num_children = self.tree[self.tree[0]+2+i]
                    tmp = sum(self.tree[idx:idx+num_children])
                    tmp_tree.append(tmp)
                    idx = idx+num_children

            for it, (out_h0, out_h1) in enumerate(zip(outs_h0, outs_h1)):
                out_h0 = F.softmax(out_h0)
                out_h1 = F.softmax(out_h1)
                global_entropy_h1 = - torch.mean(self.negative_entropy_prediction(out_h1, softmax=False))
                global_entropy_h0 = - torch.mean(self.negative_entropy_prediction(out_h0, softmax=False))

                # if self.h2_classes>1:
                #     local_entropy_h1 = - torch.mean(torch.sum(self.negative_entropy_prediction
                #                                               (out_h1, self.tree[self.tree[0]+1:], False)*h2_one_hot, 1))
                #     local_entropy_h0 = - torch.mean(torch.sum(self.negative_entropy_prediction
                #                                               (out_h0, tmp_tree, False)*h2_one_hot, 1))
                # else:
                #     local_entropy_h1 = - torch.mean(self.negative_entropy_prediction(out_h1, softmax=False))
                #     local_entropy_h0 = - torch.mean(self.negative_entropy_prediction(out_h0, softmax=False))

                out_h1_rep = out_h1.reshape(-1, per_num, self.h1_classes)
                out_h1_rep = torch.mean(out_h1_rep, 1)
                out_h0_rep = out_h0.reshape(-1, per_num, self.h0_classes)
                out_h0_rep = torch.mean(out_h0_rep, 1)
                if self.h2_classes>1:
                    local_entropy_h1_avg = torch.mean(torch.sum(self.negative_entropy_prediction
                                                                (out_h1_rep, self.tree[self.tree[0] + 1:], False)
                                                                * h2_one_hot_avg, 1))
                    local_entropy_h0_avg = torch.mean(torch.sum(self.negative_entropy_prediction
                                                                (out_h0_rep, tmp_tree, False)*h2_one_hot_avg, 1))
                else:
                    local_entropy_h1_avg = torch.mean(self.negative_entropy_prediction(out_h1_rep, softmax=False))
                    local_entropy_h0_avg = torch.mean(self.negative_entropy_prediction(out_h0_rep, softmax=False))

                instance_entropy_loss += global_entropy_h1 + global_entropy_h0
                avg_entropy_loss += local_entropy_h1_avg+local_entropy_h0_avg
        else:
            pass

        return instance_entropy_loss, avg_entropy_loss

    def calc_auxiliary_loss_h0(self, outs_h0, h0_local_labels, h1_labels, h0_labels=None, local_cls=False):
        loss = 0
        for it, (out_h0) in enumerate(outs_h0):
            if not local_cls and h0_labels is not None:
                loss += torch.mean(F.cross_entropy(out_h0, h0_labels))
            else:
                loss += self.compute_local_cls_loss(out_h0, self.tree[1:self.tree[0] + 1], self.h1_classes,
                                                    h0_local_labels, h1_labels)
        return loss

    def calc_auxiliary_loss_h1(self, outs_h1, h1_local_labels, h2_labels, h1_labels=None, local_cls=False):
        loss = 0
        for it, (out_h1) in enumerate(outs_h1):
            if not local_cls and h1_labels is not None:
                loss += torch.mean(F.cross_entropy(out_h1, h1_labels))
            elif self.h2_classes == 1:
                loss += torch.mean(F.cross_entropy(out_h1, h1_local_labels))
            else:
                loss += self.compute_local_cls_loss(out_h1, self.tree[self.tree[0]+2:], self.h2_classes,
                                                        h1_local_labels, h2_labels)
        return loss

    def calc_auxiliary_loss_h2(self, outs_h2, h2_labels):
        loss = 0
        for it, (out_h2) in enumerate(outs_h2):
            loss += torch.mean(F.cross_entropy(out_h2, h2_labels))
        return loss


##################################################################################
# Generator, qss add hierarchy nested gaussian distributions for sampling styles
##################################################################################
class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # style encoder, qss TODO 4 or 5
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)
        self.distributions = NestedDistributions(params['tree'], style_dim, params['alpha'], params['m'])

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ,
                           pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon


    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


##################################################################################
# Encoder and Decoders
##################################################################################

# qss, style dim
class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = style_dim

    def forward(self, x):
        return self.model(x)


# Tree node info
class Node(nn.Module):
    def __init__(self, miu, log_var, dist_dim, level=0, label=0, parent=-1, grad=True):
        super(Node, self).__init__()
        self.level = level
        self.label = label
        self.parent = parent
        self.miu = nn.Parameter(miu, requires_grad=grad)
        self.log_var = nn.Parameter(log_var, requires_grad=grad)
        self.dist_dim = dist_dim

    def forward(self, sample_num):
        eps = torch.randn(sample_num, self.dist_dim).cuda()
        return torch.exp(self.log_var / 2.) * eps + self.miu


# qss, nested distributions: model distributions and sample from them
class NestedDistributions(nn.Module):
    def __init__(self, tree, style_dim, threshold, margin):
        super(NestedDistributions, self).__init__()
        self.tree = tree
        self.style_dim = style_dim
        self.threshold = threshold
        # if not isinstance(margin, list):
        #     margin = [margin, margin]
        self.margin = margin

        self.root_num = tree[self.tree[0]+1]
        self.intermediate_num = tree[0]
        self.leaves_num = sum(tree[1:tree[0]+1])
        self.tree_info = nn.Sequential(*self._create_tree())

    def _create_tree(self):
        num = list([])
        num.append(self.root_num)
        num.append(self.intermediate_num)
        num.append(self.leaves_num)
        tree_info = []

        # traverse from leaf to root
        idx = 0
        level = 2
        while idx < len(self.tree):
            cur_lab = 0
            parents_num = self.tree[idx]
            for i in range(parents_num):
                children_num = self.tree[idx+i+1]
                for j in range(children_num):
                    tree_info += [Node(torch.randn(self.style_dim), torch.randn(self.style_dim), self.style_dim,
                                       level, cur_lab, sum(num[level:]) + i)]
                    cur_lab += 1
            level -= 1
            idx = idx + parents_num + 1
        # root distribution: fixed standard gaussian or trainable
        if self.root_num > 1:
            for k in range(self.root_num):
                tree_info += [Node(torch.randn(self.style_dim), torch.randn(self.style_dim), self.style_dim, 0, k, -1)]
        else:
            tree_info += [Node(torch.zeros(self.style_dim), torch.zeros(self.style_dim), self.style_dim, 0, 0, -1,
                               False)]
        return tree_info

    # fixed sample operations
    def forward(self, sample_size=100):
        batch_samples = list([])
        nodes_num = len(self.tree_info)
        for i in range(nodes_num):
            samples = self.tree_info[i].forward(sample_size)
            batch_samples.append(samples)
        return torch.cat(batch_samples, 0)

    def _kl_div(self, miu1, miu2, log_var1, log_var2):
        assert miu1.shape[0] == miu2.shape[0] and log_var1.shape[0] == log_var2.shape[0], 'dimensions of compared' \
                                                                                          ' distributions must be equal'
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)
        det1 = torch.prod(var1)
        det2 = torch.prod(var2)
        dim = miu1.shape[0]

        return 0.5*(torch.log(det2 / det1) - dim + torch.sum(torch.mul(torch.div(torch.sub(miu2, miu1), var2),
                                                        torch.sub(miu2, miu1))) + torch.sum(torch.div(var1, var2)))

    def _order_loss(self, miu1, miu2, log_var1, log_var2):
        kl_div = self._kl_div(miu1, miu2, log_var1, log_var2)
        return kl_div-self.threshold if kl_div > self.threshold else 0.

    def _is_nested(self, p, q):
        flag = False
        parent_idx = self.tree_info[p].parent
        while parent_idx != -1:
            if parent_idx == q:
                flag = True
                break
            else:
                parent_idx = self.tree_info[parent_idx].parent
        return flag

    # nested distribution loss
    def cal_dist_loss(self):
        dist_loss = 0.0
        nodes_num = len(self.tree_info)
        for i in range(nodes_num):
            miu1 = self.tree_info[i].miu
            log_var1 = self.tree_info[i].log_var
            for j in range(nodes_num):
                if i != j:
                    miu2 = self.tree_info[j].miu
                    log_var2 = self.tree_info[j].log_var
                    order_dis = self._order_loss(miu1, miu2, log_var1, log_var2)
                    if self._is_nested(i, j):
                        dist_loss += order_dis
                    else:
                        # if i >= self.leaves_num or j >= self.leaves_num:  # or is_nested(j,i)
                        dist_loss += self.margin-order_dis if order_dis < self.margin else 0.
                        # else:
                        #     dist_loss += self.margin[1]-order_dis if order_dis < self.margin[1] else 0.

        for k in range(self.root_num):
            miu1 = self.tree_info[self.leaves_num+self.intermediate_num+k].miu
            log_var1 = self.tree_info[self.leaves_num+self.intermediate_num+k].log_var
            dist_loss += self._order_loss(miu1, torch.zeros(self.style_dim).cuda(), log_var1,
                                          torch.zeros(self.style_dim).cuda())
            # order_loss = self._order_loss(miu1, log_var1, torch.zeros(self.style_dim).cuda(),
            #                               torch.zeros(self.style_dim).cuda())
            # dist_loss = dist_loss + order_dis if order_loss > 0 else dist_loss

        return dist_loss / (nodes_num * (nodes_num-1) + self.root_num)

    # compute the mahalanobis distance of the encoded style to its belonging distributions in the tree
    # Maybe make a metric learning for it
    def _mahalanobis_distance(self, s, miu, log_var):
        return torch.sum(torch.mul(torch.div(torch.sub(s, miu), torch.exp(log_var)), torch.sub(s, miu)))

    def cal_mahalanobis_loss(self, style, h0_labels, h1_labels, h2_labels):
        loss_maha = 0.
        for _, (h0_lab, h1_lab, h2_lab, s) in enumerate(zip(h0_labels, h1_labels, h2_labels, style)):
            miu_leaf = self.tree_info[h0_lab].miu
            log_var_leaf = self.tree_info[h0_lab].log_var
            miu_inter = self.tree_info[self.leaves_num+h1_lab].miu
            log_var_inter = self.tree_info[self.leaves_num+h1_lab].log_var
            miu_root = self.tree_info[self.leaves_num+self.intermediate_num+h2_lab].miu
            log_var_root = self.tree_info[self.leaves_num+self.intermediate_num+h2_lab].log_var

            loss_maha += self._mahalanobis_distance(s.squeeze(), miu_leaf, log_var_leaf) + \
                         self._mahalanobis_distance(s.squeeze(), miu_inter, log_var_inter) + \
                         self._mahalanobis_distance(s.squeeze(), miu_root, log_var_root)

        return 0.5 * loss_maha / h0_labels.size(0)

    # given a condition, sample from a particular distribution, use the tree to check the input
    def condition_sample(self, tree_level, label, sample_num=5):
        num = list([])
        num.append(self.root_num)
        num.append(self.intermediate_num)
        num.append(self.leaves_num)
        idx = sum(num[tree_level+1:]) + label
        return self.tree_info[idx].forward(sample_num)


# qss, n_downsample
class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


# qss, n_upsample
class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks, qss use layer norm ?
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer, binary output
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out  # no relu for output


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):  # padding: size to pad not value
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# Normalization layers, needs weight and bias be assigned before called
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned, i.e. gama and beta ?
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

