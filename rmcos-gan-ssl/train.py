from dataloader import dataloader
from net import generator, discriminator, classifier, initialize_weights
from utils import print_network, save_images, generate_animation, train_loss_plot, test_loss_plot, acc_plot
from loss import clf_loss, inverted_cross_entropy, d_loss, g_loss
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
import numpy as np
import pickle
import time

class GANSSL():
    def __init__(self, args):
        self.epoch = args.epoch
        self.sample_num = 100 # number images for visualize
        self.batch_size = args.batch_size
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.gan_type = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 64
        self.num_labels = args.num_labels
        self.lrC = args.lrC 
        self.m = 0.15
        self.s = 10.0
        self.alpha = 0.8
        self.device = args.device

        self.acc_time = args.acc_time_dir + "/" + args.gan_type + '_' + str(args.num_labels) + '_' + str(args.lrC) + '.txt'
        self.acc_time_best = args.acc_time_dir + "/" + args.gan_type + '_' + str(args.num_labels) + '_' + str(args.lrC) + '_best.txt'
        self.model_dir = args.model_dir + '/' + args.gan_type + '_' + str(args.num_labels) + '_' + str(args.lrC)
        self.result_dir = args.result_dir + '/' + args.gan_type + '_' + str(self.num_labels) + '_' + str(args.lrC)

        if os.path.exists(self.acc_time):   
            os.remove(self.acc_time)
        if os.path.exists(self.acc_time_best):
            os.remove(self.acc_time_best)
            
        for path in [args.acc_time_dir, self.result_dir, self.model_dir, args.labeled_data_indices]:
            if not os.path.exists(path):
                os.makedirs(path)

        #load dataset
        self.labeled_loader , self.unlabeled_loader, self.test_loader = dataloader(self.dataset,
                                                                                   args)

        #network init
        self.G = generator(input_dim=self.z_dim, output_dim=1, input_size=self.input_size).to(args.device)
        self.D = discriminator(input_dim=1, output_dim=1, input_size=self.input_size).to(args.device)
        self.C = classifier().to(args.device)

        self.G = self.G.apply(initialize_weights)
        self.D = self.D.apply(initialize_weights)
        self.C = self.C.apply(initialize_weights)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.C_optimizer = optim.SGD(self.C.parameters(), lr=args.lrC, momentum=args.momentum)
        self.C_scheduler = ReduceLROnPlateau(self.C_optimizer, "min")
        # self.C_scheduler_pretrain = StepLR(self.C_optimizer, step_size=20, gamma=0.5)
        self.G_scheduler = StepLR(self.G_optimizer, step_size=10, gamma=0.5)
        self.D_scheduler = StepLR(self.D_optimizer, step_size=10, gamma=0.5)


        # print('---------- Networks architecture -------------')
        # print_network(self.G)
        # print_network(self.D)
        # print_network(self.C)
        # print('-----------------------------------------------')

        #fixed noise
        self.sample_z_ = torch.rand(self.batch_size, self.z_dim, device=args.device)
        

    def train(self):
        train_hist = {}
        train_hist['D_loss'] = []
        train_hist['G_loss'] = []
        train_hist['C_loss'] = []
        train_hist['test_loss'] = []

        train_hist['per_epoch_time'] = []
        train_hist['total_time'] = []

        train_hist['test_accuracy'] = []

        best_acc = 0
        best_time = 0

        y_real = torch.ones(self.batch_size, 1).to(self.device)

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            print('Epoch: {}  || lrC: {}, lrD: {}, lrG: {}'.format(
                epoch + 1,
                self.C_optimizer.param_groups[0]['lr'],
                self.D_optimizer.param_groups[0]['lr'],
                self.G_optimizer.param_groups[0]['lr']))

            self.G.train()
            epoch_start_time = time.time()

            if epoch == 0:
                correct_rate = 0
                while True:
                    for iter, (x_, y_) in enumerate(self.labeled_loader):
                        x_, y_ = x_.to(self.device), y_.to(self.device)
                        self.C.train()
                        self.C_optimizer.zero_grad()
                        _, C_real = self.C(x_)
                        C_real_loss = clf_loss(C_real, y_)
                        C_real_loss.backward()
                        self.C_optimizer.step()
                        # self.C_scheduler_pretrain.step()

                        if iter == (self.labeled_loader.dataset.__len__() // self.batch_size):
                            self.C.eval()
                            test_loss = 0
                            correct = 0
                            with torch.no_grad():
                                for data, target in self.test_loader:
                                    data, target = data.to(self.device), target.to(self.device)
                                    _, output = self.C(data)
                                    test_loss += clf_loss(output, target).item() # reduction = 'mean'
                                    pred = torch.argmax(output, dim=1, keepdim=True) # get the index of the max log-probability
                                    correct += pred.eq(target.view_as(pred)).sum().item()
                            num_batch = len(self.test_loader.dataset) // self.batch_size
                            test_loss /= num_batch

                            print('Test set || Test loss: {:.4f} || Accuracy: {}/{} ({:.0f}%)\n'.format(
                                test_loss, correct, len(self.test_loader.dataset),
                                100. * correct / len(self.test_loader.dataset)
                                ))
                            correct_rate = correct / len(self.test_loader.dataset)
                            train_hist['test_accuracy'].append(correct_rate)


                    if self.num_labels == 50:
                        gate = 0.6
                    elif self.num_labels == 100:
                        gate = 0.8
                    elif self.num_labels == 600:
                        gate = 0.93
                    elif self.num_labels == 1000:
                        gate = 0.95
                    elif self.num_labels == 3000:
                        gate = 0.97
                    if correct_rate > gate:
                        break

            correct_wei = 0
            number = 0
            labeled_iter = self.labeled_loader.__iter__() # labeled_iter.__len__() = 2
            # self.labeled_loader.dataset.__len__() = 100

            self.C.train()
            for iter, (x_u, y_u) in enumerate(self.unlabeled_loader):
                if iter == self.unlabeled_loader.dataset.__len__() // self.batch_size:
                    if epoch > 0:
                        print('\nPseudo tag || Accuracy: {}/{} ({:.0f}%)\n'.format(
                            correct_wei, number,
                            100. * correct_wei / number))
                    break

                try:
                    x_l, y_l = labeled_iter.__next__()
                    # assert len(x_l) == self.batch_size, 'len of labeled and batch size must be equal'
                    if len(x_l) != self.batch_size:
                        labeled_iter = self.labeled_loader.__iter__()
                        x_l, y_l = labeled_iter.__next__()
                except StopIteration:
                    labeled_iter = self.labeled_loader.__iter__()
                    x_l, y_l = labeled_iter.__next__()

                z_ = torch.rand(self.batch_size, self.z_dim, device=self.device)
                x_l, y_l, x_u, y_u = x_l.to(self.device), y_l.to(self.device), x_u.to(self.device), y_u.to(self.device)

                # update C network
                self.C_optimizer.zero_grad()

                _, C_labeled_pred = self.C(x_l)
                C_labeled_loss = clf_loss(C_labeled_pred, y_l)

                _, C_unlabeled_pred = self.C(x_u)
                C_unlabeled_true = torch.argmax(C_unlabeled_pred, dim=1) # make labels for unlabeled data 
                C_unlabeled_loss = clf_loss(C_unlabeled_pred,  C_unlabeled_true)

                correct_wei += C_unlabeled_true.eq(y_u).sum().item()
                number += len(y_u)

                G_ = self.G(z_)
                C_fake_pred, _  = self.C(G_)
                C_fake_true = torch.argmax(C_fake_pred, dim=1) # make labels for fake data  
                C_fake_true = F.one_hot(C_fake_true, 10) # make one-hot for y true
                C_fake_loss = inverted_cross_entropy(C_fake_pred, C_fake_true)

                C_loss = C_labeled_loss + C_unlabeled_loss + C_fake_loss
           
                train_hist['C_loss'].append(C_loss.item())
                C_loss.backward()
                self.C_optimizer.step()

                # update D network
                self.D_optimizer.zero_grad()

                D_labeled = self.D(x_l)
                D_unlabeled = self.D(x_u)
                assert len(D_labeled) == len(D_unlabeled), 'length of labeled and unlabeled must be the same'
                D_real = (D_labeled + D_unlabeled)/2 

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_loss = d_loss(self.m, self.s, D_real, D_fake, y_real)

                train_hist['D_loss'].append(D_loss.item())
                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()
                z_ = torch.rand(self.batch_size, self.z_dim, device=self.device)
                G_ = self.G(z_)
                D_fake = self.D(G_)

                D_labeled = self.D(x_l)
                D_unlabeled = self.D(x_u)
                assert len(D_labeled) == len(D_unlabeled), 'length of labeled and unlabeled must be equal'
                D_real = (D_labeled + D_unlabeled)/2 # xem xet dieu chinh !!!

                G_loss_D = g_loss(self.m, self.s, D_real, D_fake, y_real)

                _, C_fake_pred = self.C(G_)
                C_fake_true = torch.argmax(C_fake_pred, dim=1) #(vals, indices) -> take indices
                G_loss_C  = clf_loss(C_fake_pred, C_fake_true)

                G_loss = self.alpha * G_loss_D + (1-self.alpha) * G_loss_C # xem xet dieu chinh

                train_hist['G_loss'].append(G_loss.item())
                G_loss_D.backward(retain_graph=True)
                G_loss_C.backward()

                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: {:3d} || {}/{} || D_loss: {:.4f}, G_loss: {:.4f}, C_loss: {:.4f}".format(
                        epoch+1, iter+1, self.unlabeled_loader.dataset.__len__() // self.batch_size,
                        D_loss.item(),
                        G_loss.item(),
                        C_loss.item())
                    )

            self.C.eval()
            average_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    _, output = self.C(data)
                    test_loss = clf_loss(output, target).item()  # reduction = 'mean'
                    train_hist['test_loss'].append(test_loss)
                    pred = torch.argmax(output, dim=1, keepdim=True)  # get the index of the max log-probability
                    average_loss += test_loss
                    correct += pred.eq(target.view_as(pred)).sum().item()

            num_batch = len(self.test_loader.dataset) // self.batch_size
            average_loss /= num_batch
            # train_hist['test_loss'].append(test_loss)

            correct_rate = correct / len(self.test_loader.dataset)
            cur_time = time.time() - epoch_start_time
            with open(self.acc_time, 'a') as f:
                f.write(str(cur_time) + ' ' + str(correct_rate) + '\n')

            if correct_rate > best_acc:
                best_acc = correct_rate
                best_time = cur_time

            print('\nTest set || Test loss: {:.4f} || Accuracy: {}/{} ({:.4f}%)\n'.format(
                average_loss, correct, len(self.test_loader.dataset),
                100. * correct_rate))
            
            train_hist['test_accuracy'].append(correct_rate)
            train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            with torch.no_grad():
                self.visualize_results((epoch + 1))
            
            self.C_scheduler.step(average_loss)
            self.G_scheduler.step()
            self.D_scheduler.step()

        with open(self.acc_time_best, 'a') as f:
            f.write(str(best_time) + ' ' + str(best_acc) + '\n')

        train_hist['total_time'].append(time.time() - start_time)
        print("Average one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(train_hist['per_epoch_time']),
                                                                        self.epoch,
                                                                        train_hist['total_time'][0]))
        self.train_hist = train_hist
        # save model C, G, D
        self.save()

        # make animation image (gif)
        generate_animation(self.result_dir, self.epoch)
        # save loss img
        train_loss_plot(self.train_hist, self.result_dir)
        test_loss_plot(self.train_hist, self.result_dir)
        # save acc img
        acc_plot(self.train_hist, self.result_dir)
    
    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim)).to(self.device)
            samples = self.G(sample_z_)

        if self.device == 'cuda':
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          os.path.join(self.result_dir, 'epoch%03d' % epoch + '.png'))

    def save(self):
        torch.save(self.G.state_dict(), os.path.join(self.model_dir, 'G.pkl'))
        torch.save(self.C.state_dict(), os.path.join(self.model_dir, 'C.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.model_dir, 'D.pkl'))

        with open(os.path.join(self.model_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        self.G.load_state_dict(torch.load(os.path.join(self.model_dir, 'G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(self.model_dir, 'D.pkl')))
        self.C.load_state_dict(torch.load(os.path.join(self.model_dir, 'C.pkl')))
