from networks import AdaINGen, MsImageDis
from utils import weights_init, get_model_list, get_scheduler
import torch
import torch.nn as nn
import os
from data import get_super_category, scale_local_label


class HIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(HIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen = AdaINGen(hyperparameters['input_dim'], hyperparameters['gen'])  # auto-encoder
        self.dis = MsImageDis(hyperparameters['input_dim'], hyperparameters['dis'])  # discriminator

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis.parameters())
        gen_params = list(self.gen.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Applies ``fn`` recursively to every submodule Network weight initialization, gen use kaiming, dis use gaussian
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

        if 'local_cls' in hyperparameters.keys():
            self.local_cls = hyperparameters['local_cls']
        else:
            self.local_cls = False
        if 'style_adv' in hyperparameters.keys():
            self.style_adv = hyperparameters['style_adv']
        else:
            self.style_adv = False

        if 'cycle_entropy' in hyperparameters.keys():
            self.cycle = hyperparameters['cycle_entropy']
        else:
            self.cycle = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def total_variation_loss(self, x):
        assert x.ndimension() == 4

        a = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        b = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

        return a+b

    # style transfer
    def forward(self, x1, x2):
        self.eval()
        c1, s1 = self.gen.encode(x1.unsqueeze(0))
        c2, s2 = self.gen.encode(x2.unsqueeze(0))
        x_ab = self.gen.decode(c1, s2)
        x_ba = self.gen.decode(c2, s1)
        self.train()
        return x_ab, x_ba

    # randomly sample styles from one tree node
    def style_rand_sampling(self, num, per_num=1, level=None, labels=None):
        # whether train the root with only one node in level 0 ? qss
        train_root = True
        if self.dis.h2_classes < 2 and train_root is False:
            low = 1
        else:
            low = 0
        if level is None:
            level = torch.randint(low=low, high=3, size=(1,), dtype=torch.int32).item()
        s_list = list([])
        if level == 0:
            nodes_num = self.gen.distributions.root_num
        elif level == 1:
            nodes_num = self.gen.distributions.intermediate_num
        else:
            nodes_num = self.gen.distributions.leaves_num
        if labels is None:
            labels = torch.randint(low=0, high=nodes_num, size=(num,1), dtype=torch.long)
        for i in range(num):
            s_list.append(self.gen.distributions.condition_sample(level, labels[i].item(), sample_num=per_num))
        labels = labels.repeat(1, per_num)
        labels = labels.reshape(-1, 1).squeeze()

        return torch.cat(s_list, dim=0), level, labels

    # qss
    def gen_update(self, x, hyperparameters, h0_label, h0_local_label, h1_label, h1_local_label, h2_label):
        self.gen_opt.zero_grad()

        # encode
        c, s_prime = self.gen.encode(x)

        # decode (within domain), use style from real image
        x_recon = self.gen.decode(c, s_prime)

        per_num = 5 if self.cycle else 1
        s_dist, level, labels = self.style_rand_sampling(x.size(0), per_num=per_num)
        # decode (cross domain), use style from a distribution
        shape = c.shape
        c = c.repeat(1, per_num, 1, 1)
        c = c.reshape(-1, shape[1], shape[2], shape[3])
        x_trans = self.gen.decode(c, s_dist)

        # encode again
        c_recon, s_dist_recon = self.gen.encode(x_trans)

        # reconstruction loss, qss
        self.loss_gen_recon_x = self.recon_criterion(x_recon, x)
        self.loss_gen_recon_c = self.recon_criterion(c, c_recon)
        self.loss_gen_recon_s = self.recon_criterion(s_dist, s_dist_recon)

        self.loss_gen_tv = self.total_variation_loss(x_trans) \
            if 'tv_w' in hyperparameters.keys() and hyperparameters['tv_w'] > 0 else 0.

        # distributions nest loss
        self.loss_gen_dist_nest = self.gen.distributions.cal_dist_loss() \
            if 'dist_w' in hyperparameters.keys() and hyperparameters['dist_w'] >0 else 0.
        self.loss_gen_maha_dist = self.gen.distributions.cal_mahalanobis_loss(s_prime, h0_label, h1_label, h2_label) \
            if 'maha_w' in hyperparameters.keys() and hyperparameters['maha_w'] > 0 else 0.

        # qss adversarial training of disentangled styles
        if self.style_adv:
            self.loss_gen_adv_style, self.loss_gen_cls_h0_style = self.dis.calc_gen_style_loss(s_prime, h0_label.cuda())
        else:
            self.loss_gen_adv_style, self.loss_gen_cls_h0_style = 0., 0.

        # GAN loss
        if level == 0:
            h2_labels = labels
            if self.cycle:
                self.loss_gen_adv, self.loss_gen_cls_h0, self.loss_gen_cls_h1, self.loss_gen_cls_h2, \
                self.loss_gen_instance_entropy, self.loss_gen_avg_entropy = \
                    self.dis.calc_gen_loss(x_trans, h2_labels=h2_labels.cuda(), local_cls=self.local_cls,
                                           per_num=per_num, entropy_h0=True, entropy_h1=True)
            else:
                self.loss_gen_adv, self.loss_gen_cls_h0, self.loss_gen_cls_h1, self.loss_gen_cls_h2,\
                    self.loss_gen_instance_entropy, self.loss_gen_avg_entropy = \
                    self.dis.calc_gen_loss(x_trans, h2_labels=h2_labels.cuda(), local_cls=self.local_cls)

        elif level == 1:
            h1_labels = labels
            h1_local_labels = scale_local_label(h1_labels, self.gen.distributions.tree[self.gen.distributions.tree[0]+1:])
            h1_local_labels = torch.from_numpy(h1_local_labels)
            h2_labels = get_super_category(h1_labels, self.gen.distributions.tree[self.gen.distributions.tree[0] + 1:])
            h2_labels = torch.from_numpy(h2_labels)
            if self.cycle:
                self.loss_gen_adv, self.loss_gen_cls_h0, self.loss_gen_cls_h1, self.loss_gen_cls_h2, \
                self.loss_gen_instance_entropy, self.loss_gen_avg_entropy = \
                    self.dis.calc_gen_loss(x_trans, h1_labels=h1_labels.cuda(), h1_local_labels=h1_local_labels.cuda(),
                                           h2_labels=h2_labels.cuda(), local_cls=self.local_cls, per_num=per_num,
                                           entropy_h0=True)
            else:
                self.loss_gen_adv, self.loss_gen_cls_h0, self.loss_gen_cls_h1, self.loss_gen_cls_h2,\
                    self.loss_gen_instance_entropy, self.loss_gen_avg_entropy = \
                    self.dis.calc_gen_loss(x_trans, h1_labels=h1_labels.cuda(), h1_local_labels=h1_local_labels.cuda(),
                                           h2_labels=h2_labels.cuda(), local_cls=self.local_cls)

        else:
            h0_labels = labels
            h0_local_labels = scale_local_label(h0_labels, self.gen.distributions.tree)
            h0_local_labels = torch.from_numpy(h0_local_labels)
            h1_labels = get_super_category(h0_labels, self.gen.distributions.tree)
            h1_local_labels = scale_local_label(h1_labels, self.gen.distributions.tree[self.gen.distributions.tree[0]+1:])
            h1_local_labels = torch.from_numpy(h1_local_labels)
            h2_labels = get_super_category(h1_labels, self.gen.distributions.tree[self.gen.distributions.tree[0] + 1:])
            h1_labels = torch.from_numpy(h1_labels)
            h2_labels = torch.from_numpy(h2_labels)
            self.loss_gen_adv, self.loss_gen_cls_h0, self.loss_gen_cls_h1, self.loss_gen_cls_h2, \
            self.loss_gen_instance_entropy, self.loss_gen_avg_entropy = \
                self.dis.calc_gen_loss(x_trans, h0_labels=h0_labels.cuda(), h0_local_labels=h0_local_labels.cuda(),
                                       h1_labels=h1_labels.cuda(), h1_local_labels=h1_local_labels.cuda(),
                                       h2_labels=h2_labels.cuda(), local_cls=self.local_cls)

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * \
            (self.loss_gen_adv + self.loss_gen_cls_h0 + self.loss_gen_cls_h1 + self.loss_gen_cls_h2) + \
            hyperparameters['dist_w'] * self.loss_gen_dist_nest + \
            hyperparameters['maha_w'] * self.loss_gen_maha_dist + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x + \
            hyperparameters['recon_s_w'] * self.loss_gen_recon_s + \
            hyperparameters['recon_c_w'] * self.loss_gen_recon_c + \
            hyperparameters['tv_w'] * self.loss_gen_tv + \
            hyperparameters['gan_w'] * (self.loss_gen_adv_style + self.loss_gen_cls_h0_style) + \
            hyperparameters['gan_w'] * (self.loss_gen_instance_entropy + self.loss_gen_avg_entropy)

        self.loss_gen_total.backward()
        self.gen_opt.step()


    # qss, recon & sampled styles to transfer x to several nodes of the tree
    def rand_trans(self, x):
        self.eval()

        sample_num_per_node = 1
        s_dist, level, labels = self.style_rand_sampling(x.size(0), sample_num_per_node)
        x_recon, x_trans = [], []
        for i in range(x.size(0)):
            c, s_fake = self.gen.encode(x[i].unsqueeze(0))
            x_recon.append(self.gen.decode(c, s_fake))
            x_trans.append(self.gen.decode(c, s_dist[i].unsqueeze(0)))

        self.train()
        return torch.cat(x_recon), torch.cat(x_trans), level, labels, sample_num_per_node

    # qss, single image conditional translation
    def conditional_trans(self, x, level, label, sample_num):
        self.eval()
        c, s_prime = self.gen.encode(x.unsqueeze(0))
        x_recon = self.gen.decode(c, s_prime)
        s_dist = self.gen.distributions.condition_sample(level, label, sample_num)
        c_sets = c.repeat(sample_num, 1, 1, 1)
        x_trans = self.gen.decode(c_sets, s_dist)
        self.train()

        return x_recon, x_trans

    # qss, h0, h1 labels needed, only one images data, both dis loss & auxiliary loss
    def dis_update(self, x,  hyperparameters, h0_label, h0_local_label, h1_label, h1_local_label, h2_label):
        self.dis_opt.zero_grad()
        # qss adversarial training of disentangled styles and sampled distributions' styles
        labels = torch.randint(low=0, high=self.dis.h0_classes, size=(x.size(0),), dtype=torch.long)
        s_list = list([])
        level = 2
        for i in range(x.size(0)):
            s_list.append(self.gen.distributions.condition_sample(level, labels[i].item(), sample_num=1))
        s_list = torch.cat(s_list, 0)

        # encode
        c, s_input = self.gen.encode(x)

        # decode (cross domain), cross generation only uses sampled style
        if not self.dis.multi_adv:
            s_dist, _, _ = self.style_rand_sampling(x.size(0))
            x_trans = self.gen.decode(c, s_dist)

            # D loss, detach for no grad for input fake images
            self.loss_dis_adv, self.loss_cls_h0, self.loss_cls_h1, self.loss_clc_h2 = \
                self.dis.calc_dis_loss(x_trans.detach(), x, h0_label, h0_local_label, h1_label, h1_local_label, h2_label,
                                       self.local_cls)
        else:
            # level 0, qss, select one from the three
            # h2_labels = None
            h2_labels = h2_label.clone()
            x_rnd_idx = torch.randperm(x.size(0))
            h2_labels = h2_labels[x_rnd_idx]

            s_dist_l0, _, h2_labels_fake_l0 = self.style_rand_sampling(x.size(0), level=0, labels=h2_labels)
            x_trans_l0 = self.gen.decode(c, s_dist_l0)

            # D loss, detach for no grad for input fake images
            loss_dis_adv_l0, loss_cls_h0_l0, loss_cls_h1_l0, loss_clc_h2_l0 = \
                self.dis.calc_dis_loss(x_trans_l0.detach(), x, h2_labels=h2_label, local_cls=self.local_cls,
                                       h2_labels_fake=h2_labels_fake_l0.cuda())

            # level 1, qss
            # h1_labels = None
            h1_labels = h1_label.clone()
            h1_labels = h1_labels[x_rnd_idx]

            s_dist_l1, _, h1_labels_fake_l1 = self.style_rand_sampling(x.size(0), level=1, labels=h1_labels)
            x_trans_l1 = self.gen.decode(c, s_dist_l1)
            h1_local_labels_fake_l1 = scale_local_label(h1_labels_fake_l1.cpu(),
                                                        self.gen.distributions.tree[self.gen.distributions.tree[0] + 1:])
            h1_local_labels_fake_l1 = torch.from_numpy(h1_local_labels_fake_l1)
            h2_labels_fake_l1 = get_super_category(h1_labels_fake_l1.cpu(),
                                                   self.gen.distributions.tree[self.gen.distributions.tree[0] + 1:])
            h2_labels_fake_l1 = torch.from_numpy(h2_labels_fake_l1)

            # D loss, detach for no grad for input fake images
            loss_dis_adv_l1, loss_cls_h0_l1, loss_cls_h1_l1, loss_clc_h2_l1 = \
                self.dis.calc_dis_loss(x_trans_l1.detach(), x, h1_labels=h1_label, h1_local_labels=h1_local_label,
                                       h2_labels=h2_label, local_cls=self.local_cls, h2_labels_fake=h2_labels_fake_l1.cuda(),
                                       h1_local_labels_fake=h1_local_labels_fake_l1.cuda(),
                                       h1_labels_fake=h1_labels_fake_l1.cuda())

            # level 2, qss
            # h0_labels = None
            h0_labels = h0_label.clone()
            h0_labels = h0_labels[x_rnd_idx]

            s_dist_l2, _, h0_labels_fake_l2 = self.style_rand_sampling(x.size(0), level=2, labels=h0_labels)
            x_trans_l2 = self.gen.decode(c, s_dist_l2)
            h0_local_labels_fake_l2 = scale_local_label(h0_labels_fake_l2.cpu(), self.gen.distributions.tree)
            h0_local_labels_fake_l2 = torch.from_numpy(h0_local_labels_fake_l2)
            h1_labels_fake_l2 = get_super_category(h0_labels_fake_l2.cpu(), self.gen.distributions.tree)
            h1_local_labels_fake_l2 = scale_local_label(h1_labels_fake_l2,
                                                        self.gen.distributions.tree[self.gen.distributions.tree[0] + 1:])
            h1_local_labels_fake_l2 = torch.from_numpy(h1_local_labels_fake_l2)
            h2_labels_fake_l2 = get_super_category(h1_labels_fake_l2,
                                                   self.gen.distributions.tree[self.gen.distributions.tree[0] + 1:])
            h1_labels_fake_l2 = torch.from_numpy(h1_labels_fake_l2)
            h2_labels_fake_l2 = torch.from_numpy(h2_labels_fake_l2)

            # D loss, detach for no grad for input fake images
            loss_dis_adv_l2, loss_cls_h0_l2, loss_cls_h1_l2, loss_clc_h2_l2 = \
                self.dis.calc_dis_loss(x_trans_l2.detach(), x, h0_label, h0_local_label, h1_label, h1_local_label,
                                       h2_label, self.local_cls, h2_labels_fake=h2_labels_fake_l2.cuda(),
                                       h1_local_labels_fake=h1_local_labels_fake_l2.cuda(),
                                       h1_labels_fake=h1_labels_fake_l2.cuda(),
                                       h0_local_labels_fake=h0_local_labels_fake_l2.cuda(),
                                       h0_labels_fake=h0_labels_fake_l2.cuda())

            self.loss_dis_adv = loss_dis_adv_l0 + loss_dis_adv_l1 + loss_dis_adv_l2
            self.loss_cls_h0 = loss_cls_h0_l0 + loss_cls_h0_l1 + loss_cls_h0_l2
            self.loss_cls_h1 = loss_cls_h1_l0 + loss_cls_h1_l1 + loss_cls_h1_l2
            self.loss_clc_h2 = loss_clc_h2_l0 + loss_clc_h2_l1 + loss_clc_h2_l2

        if self.style_adv:
            self.loss_dis_adv_style, self.loss_cls_h0_style = self.dis.calc_dis_style_loss(s_list.detach(),
                                                                                       s_input.detach(), labels.cuda())
        else:
            self.loss_dis_adv_style, self.loss_cls_h0_style = 0., 0.
        self.loss_dis = hyperparameters['gan_w'] * \
            (self.loss_dis_adv + self.loss_cls_h0 + self.loss_cls_h1 + self.loss_clc_h2) \
             + hyperparameters['gan_w'] * (self.loss_dis_adv_style + self.loss_cls_h0_style)
        self.loss_dis.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    # qss, gen, dis model
    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict['dis'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    # qss, gen, dis model
    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen': self.gen.state_dict()}, gen_name)
        torch.save({'dis': self.dis.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
