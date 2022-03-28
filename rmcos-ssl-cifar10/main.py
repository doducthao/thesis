import os
import time
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from parser import create_parser
from datasets import cifar10
from dataloader import NO_LABEL, relabel_dataset, TwoStreamBatchSampler
import networks
from networks import generator, discriminator

from losses import (BCEloss, inverted_cross_entropy, g_loss, d_loss,
    softmax_mse_loss, softmax_kl_loss, symmetric_mse_loss)

from utils import AverageMeterSet, assert_exactly_one, save_images

from ramps import linear_rampup, cosine_rampdown, sigmoid_rampup

def create_model(args, ema=False):
    LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
        pretrained='pre-trained' if args.pretrained else '',
        ema='EMA ' if ema else '',
        arch=args.arch))

    model_factory = networks.__dict__[args.arch]
    model_params = dict(pretrained=args.pretrained, num_classes=10)
    model = model_factory(**model_params)

    if ema:  # # exponential moving average
        for param in model.parameters():
            param.detach_()
    return model

def load_checkpoint(args, model, ema_model, G, D, optimizer):
    LOG.info("=> loading checkpoint '{}'".format(args.resume))
    best_file = os.path.join(args.resume, 'best.ckpt')
    G_file = os.path.join(args.resume, 'G.pkl')
    C_file = os.path.join(args.resume, 'D.pkl')

    assert os.path.isfile(best_file), "=> no checkpoint found at '{}'".format(best_file)
    assert os.path.isfile(G_file), "=> no checkpoint found at '{}'".format(G_file)
    assert os.path.isfile(C_file), "=> no checkpoint found at '{}'".format(C_file)

    checkpoint = torch.load(best_file)
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    G.load_state_dict(torch.load(G_file))
    D.load_state_dict(torch.load(C_file))

    print('----------------best_precl----------------', best_prec1)

    LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    return global_step, best_prec1, start_epoch

def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = relabel_dataset(dataset, labels)

    if args.exclude_unlabeled: # False
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size: ## True
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
            # unlabeled: 97, labeled: 31 -> batch_size: 128 
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    # tao train loader voi 31 labeled va 97 unlabeled
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)

    return train_loader, eval_loader

def save_checkpoint(state, dirpath):
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, best_path)


def adjust_learning_rate(args, optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    lr = linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_current_consistency_weight(args, epoch):
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def generated_weight(epoch):
    alpha = 0.0
    T1 = 10
    T2 = 60
    af = 0.3
    if epoch > T1:
        alpha = (epoch-T1) / (T2-T1)*af
        if epoch > T2:
            alpha = af
    return alpha

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size.float()))
    return res
    
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1-alpha, param.data)
        ema_param.data *= alpha
        ema_param.data += (1-alpha) * param.data


def visualize_results(args, G, epoch, checkpoint_path):
    G.eval()
    generated_images_dir = os.path.join(checkpoint_path, "generated_images")
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)

    tot_num_samples = 64
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    sample_z_ = torch.rand((tot_num_samples, args.z_dim)).cuda()

    samples = G(sample_z_)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)

    save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                      generated_images_dir + '/' + 'epoch%03d' % epoch + '.png')
                          

def validate(args, eval_loader, model):
    class_criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (input, target) in enumerate(eval_loader):
        with torch.no_grad():
            input = torch.autograd.Variable(input.cuda())
            target = torch.autograd.Variable(target.cuda())

        minibatch_size = len(target)
        labeled_minibatch_size = target.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        # compute output
        output, _ = model(input)
        class_loss = class_criterion(output, target) / minibatch_size

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target.data, topk=(1,))
        meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                "Test: [{}/{}] |\
                batch_time: {:.3f} |\
                class_loss: {:.4f} |\
                top1 {:.3f}".format(
                    i,
                    len(eval_loader),
                    meters["batch_time"].val,
                    meters["class_loss"].val,
                    meters["top1"].val.item()))

    LOG.info(' *Top1 {:.3f}'.format(meters["top1"].avg.item()))

    return meters['top1'].avg

def train(args, checkpoint_path, train_loader, model, ema_model, optimizer, G, D, G_optimizer, D_optimizer, epoch):
    global global_step

    class_criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=NO_LABEL).cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    else:
        assert False, args.consistency_type

    residual_logit_criterion = symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()
    D.train()
    G.train()

    end = time.time()

    for i, ((input, ema_input), target) in enumerate(train_loader):
        adjust_learning_rate(args, optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        input = torch.autograd.Variable(input.cuda())

        with torch.no_grad():
            ema_input = torch.autograd.Variable(ema_input.cuda())

        target = torch.autograd.Variable(target.cuda())

        minibatch_size = len(target)
        labeled_minibatch_size = target.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        ema_model_out = ema_model(ema_input) # tuple shape (128, 10), (128, 10 )
        model_out = model(input)

        assert len(model_out) == 2
        assert len(ema_model_out) == 2

        class_logit, cons_logit = model_out
        ema_logit, _ = ema_model_out
        
        del ema_input, model_out, ema_model_out

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

        if args.logit_distance_cost >= 0: # 0.01
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            meters.update('res_loss', res_loss.item())

        else:
            class_logit, cons_logit = class_logit, class_logit
            res_loss = 0

        class_loss = class_criterion(class_logit, target) / minibatch_size
        meters.update('class_loss', class_loss.item())

        # ema_class_loss = class_criterion(ema_logit, target) / minibatch_size
        # meters.update('ema_class_loss', ema_class_loss.item())

        if args.consistency: # 100
            consistency_weight = get_current_consistency_weight(args, epoch)
            meters.update('cons_weight', consistency_weight)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size

            meters.update('cons_loss', consistency_loss.item())
        else:
            consistency_loss = 0
            meters.update('cons_loss', consistency_loss)

        z_ = torch.rand((args.generated_batch_size, args.z_dim))
        z_ = z_.cuda()
        G_ = G(z_)

        C_fake_pred, _ = model(G_)
        C_fake_pred = F.softmax(C_fake_pred, dim=1)

        with torch.no_grad():
            C_fake_wei = torch.max(C_fake_pred, 1)[1]
            C_fake_wei = C_fake_wei.view(-1, 1)
            C_fake_wei = torch.zeros(args.generated_batch_size, 10).cuda().scatter_(1, C_fake_wei, 1)

        # C_fake_loss = nll_loss_neg(C_fake_pred, C_fake_wei)
        C_fake_loss = inverted_cross_entropy(C_fake_pred, C_fake_wei)

        loss = class_loss + consistency_loss + res_loss + generated_weight(epoch) * C_fake_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        prec1 = accuracy(class_logit.data, target.data, topk=(1,))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)

        ema_prec1 = accuracy(ema_logit.data, target.data, topk=(1,))
        meters.update('ema_top1', ema_prec1[0], labeled_minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            LOG.info(
                "Epoch: [{}/{}][{}/{}] | batch_time: {:.3f} | class_loss: {:.4f} | cons_loss: {:.4f} | top1: {:.3f}".format(
                    epoch,
                    args.epochs,
                    i,
                    len(train_loader),
                    meters["batch_time"].val,
                    meters["class_loss"].val,
                    meters["cons_loss"].val,
                    meters["top1"].val.item()
                    ))

        # update D network
        D_optimizer.zero_grad()
        G_ = G(z_)

        D_real = D(input) # (128, 1)
        D_fake = D(G_) # (32, 1)
        D_loss = d_loss(D_real, D_fake, torch.ones_like(D_real))

        D_loss.backward()
        D_optimizer.step()

        # update G network
        G_optimizer.zero_grad()
        G_ = G(z_)

        D_real = D(input)
        D_fake = D(G_)

        G_loss_D = g_loss(D_real, D_fake, torch.ones_like(D_fake))

        C_fake_pred, _ = model(G_)
        C_fake_pred = F.log_softmax(C_fake_pred, dim=1)

        with torch.no_grad():
            C_fake_wei = torch.max(C_fake_pred, 1)[1]

        G_loss_C = F.nll_loss(C_fake_pred, C_fake_wei)

        G_loss = G_loss_D + generated_weight(epoch) * G_loss_C
        if epoch <= 10:
            G_loss_D.backward()
        else:
            G_loss_D.backward(retain_graph=True)
            G_loss_C.backward()

        G_optimizer.step()

        if i % args.print_freq == 0:
            LOG.info("Epoch: [{}/{}][{}/{}] | D_loss: {:.6f} | G_loss: {:.6f}".format(
                    epoch,
                    args.epochs,
                    i,
                    len(train_loader),
                    D_loss.item(),
                    G_loss.item()))

    with torch.no_grad():
        visualize_results(args, G, (epoch + 1), checkpoint_path)

if __name__ == "__main__":
    args = create_parser()
    if args.resume:
        checkpoint_path = args.resume
        date_time_now = datetime.now()
        date_time_now = "{:%Y-%m-%d_%H:%M:%S}".format(date_time_now)
    else:
        date_time_now = datetime.now()
        date_time_now = "{:%Y-%m-%d_%H:%M:%S}".format(date_time_now)
        checkpoint_path = os.path.join("out", args.dataset, date_time_now)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(checkpoint_path, "logging.log"),
        filemode="a")
    
    LOG = logging.getLogger("main")
    LOG.info("-"*50)
    LOG.info("Start training at {}".format(date_time_now))

    # prepare datasets
    dataset_config = cifar10()
    train_transformation = dataset_config["train_transformation"]
    eval_transformation = dataset_config["eval_transformation"]
    datadir = dataset_config["datadir"]
    num_classes = dataset_config["num_classes"]
    train_loader, eval_loader = create_data_loaders(train_transformation,
                                                    eval_transformation,
                                                    datadir,
                                                    args)
    # create models for classify
    model = create_model(args).cuda()
    ema_model = create_model(args, ema=True).cuda()

    # create generator and discriminator
    G = generator(input_dim=args.z_dim, output_dim=3, input_size=32).cuda()
    D = discriminator(input_dim=3, output_dim=1, input_size=32).cuda()

    # create optimizers
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=args.nesterov)

    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

    # load checkpoints if need
    if args.resume:
        global_step, best_prec1, args.start_epoch = load_checkpoint(args, model, ema_model, G, D, optimizer)
    else:
        global_step, best_prec1 = 0, 0

    is_best = False
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train(args, checkpoint_path, train_loader, model, ema_model, optimizer, G, D, G_optimizer, D_optimizer, epoch)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(args, eval_loader, model)
            LOG.info("Evaluating the EMA model:")
            ema_prec1 = validate(args, eval_loader, ema_model)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0 and is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()},
                checkpoint_path)
            torch.save(G.state_dict(), os.path.join(checkpoint_path, 'G.pkl'))
            torch.save(D.state_dict(), os.path.join(checkpoint_path, 'D.pkl'))
    

    
        



