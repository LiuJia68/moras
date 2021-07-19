import sys
# update your projecty root path before running
sys.path.append('../..')

import os
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from models.vgg import *
from models.macro_models import EvoNetwork
# from models.micro_models import NetworkCIFAR as Network
from models.micro_models import NetworkCIFAR_Jia as Network

import torch.nn.functional as F
from apex import amp
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, evaluate_various_atk)

# import search.cifar10_search as my_cifar10
from torchvision import transforms, datasets
import time
from misc import utils
from search import micro_encoding
from search import macro_encoding
from misc.flops_counter import add_flops_counting_methods

import torchattacks

logger = logging.getLogger(__name__)

device = 'cuda'
# device = 'cuda:0'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()

def main(genome, index=0, save='Design_1', expr_root='search', seed=0, gpu=0, init_channels=24,
         layers=11, auxiliary=False, cutout=False, drop_path_prob=0.0):

    # ---- train logger ----------------- #
    save_pth = os.path.join(expr_root, '{}'.format(save))
    utils.create_exp_dir(save_pth)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(save_pth, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

    CIFAR_CLASSES = 10

    genotype = micro_encoding.decode(genome)
    # model = Network(init_channels, CIFAR_CLASSES, layers, auxiliary, genotype)
    model = Network(init_channels, CIFAR_CLASSES, layers, genotype)

    # logging.info("Genome = %s", genome)
    logging.info("Architecture = %s", genotype)

    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)

    n_params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, model.parameters())) / 1e6)
    model = model.to(device)
    logging.info("param size = %fMB", n_params)

    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    # logfile = os.path.join(args.out_dir, 'output_{}.log'.format(ind)) #Jia modified
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))
    logger.info(args)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    # model = PreActResNet18().cuda()
    # model = net.cuda() #Jia modified in 7 Oct 2020
    model.droprate = 0.0
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))

    # for epoch in range(epochs):
    #     scheduler.step()
    #     logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    #     model.droprate = drop_path_prob * epoch / epochs
    # train_acc, net = train(train_loader, model, criterion, optimizer, train_params, epochs, save_pth, drop_path_prob)
    train_acc, net = ad_tr(train_loader, model, criterion, args.epochs, scheduler, args.delta_init, epsilon, pgd_alpha, alpha, opt, args.early_stop, args.out_dir, save_pth)
    # torch.save(obj=net.state_dict(), f="{}/{}.pth".format(save_pth, save))
    # net.aux_logits = False
    # logging.info('train_acc %f', train_acc)
    print("-" * 50)
    # random whitebox attacks
    net.eval()
    net.half()
    r = np.random.randint(0, high=5)
    # r = 0
    atk_acc, attack_type = rdm_atk(test_loader, net, r)
    print(r)

    # black-box attack based on FGSM VGG-16
    # fgsm_attack = FGSM(net)
    # fgsm_attack.save(data_loader=test_queue, file_name="data/cifar_fgsm.pt", accuracy=True)
    # attack_type = 'Black-box_FGSM'
    # fgsm_attack = torchattacks.FGSM(net, eps=0.007)
    # adv_data_fgsm = fgsm_attack.load(file_name="../data/cifar10_fgsm.pt", scale=True)
    # adv_loader_fgsm = torch.utils.data.DataLoader(adv_data_fgsm, batch_size=batch_size,
    #                                               shuffle=True, num_workers=0)
    # atk_acc = black_vgg_fgsm(adv_loader_fgsm, net, criterion)
    # valid_acc = infer(test_loader, net, criterion)
    valid_acc = new_infer(test_loader, net)
    # print('The {}th net'.format(index))
    # logger.info(' Test Acc \t atk acc \t atc type')
    # # logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
    # logger.info(' \t %.4f \t %.4f \t %.4s', valid_acc, atk_acc, attack_type)

    # calculate for flops
    model = add_flops_counting_methods(net)
    model.eval()
    model.start_flops_count()
    random_data = torch.randn(1, 3, 32, 32)
    model(torch.autograd.Variable(random_data).to(device))
    n_flops = np.round(model.compute_average_flops_cost() / 1e6, 4)

    logging.info('train_acc %.4f valid_acc %.4f atk_acc %.4f flops = %f attck_type = %s', train_acc, valid_acc, atk_acc, n_flops, attack_type)

    # save to file
    # os.remove(os.path.join(save_pth, 'log.txt'))
    with open(os.path.join(save_pth, 'log.txt'), "w") as file:
        file.write("Genome = {}\n".format(genome))
        file.write("Architecture = {}\n".format(genotype))
        file.write("param size = {} MB\n".format(n_params))
        file.write("flops = {} MB\n".format(n_flops))
        file.write("valid_acc = {}\n".format(valid_acc))
        file.write("valid_acc_atk = {}\n".format(atk_acc))
        file.write("atk_type = {}\n".format(attack_type))
    # logging.info("Architecture = %s", genotype))
    return {
        'valid_acc': valid_acc,
        'valid_acc_atk': atk_acc,
        'params': n_params,
        'flops': n_flops,
        'atk_type': r
    }


def rdm_atk(test_queue, model, type):

    # attacks_name = ['FGSM', 'DeepFool','BIM', 'CW', 'RFGSM', 'PGD','FFGSM', 'TPGD', 'MultiAttack']

    attacks_name = ['FGSM', 'BIM', 'PGD', 'FFGSM', 'Blk']
    # attacks_name = ['FGSM', 'MultiAttack']
    print("Attack Image & Predicted Label")

    # print(attack)
    if type < 4:
        attacks = [
            torchattacks.FGSM(model, eps=8 / 255),
            # torchattacks.DeepFool(net, steps=1),
            torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=7),
            # torchattacks.CW(net, c=1, kappa=0, steps=10, lr=0.01),
            # torchattacks.RFGSM(model, eps=8 / 255, alpha=4 / 255, steps=1),
            torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=7),
            torchattacks.FFGSM(model, eps=8 / 255, alpha=12 / 255),
            # torchattacks.TPGD(model, eps=8 / 255, alpha=2 / 255, steps=1)
            # torchattacks.MultiAttack(model, [
            #     torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=1, random_start=True)] * 2),
        ]
        atk_loss = 0
        atk_acc = 0
        n = 0
        model.eval()

        for i, (X, y) in enumerate(test_queue):
            X, y = X.cuda(), y.cuda()
            images = attacks[type](X, y)
            # labels = labels.cuda()
            # labels = labels.to(device)
            with torch.no_grad():
                output = model(images)
                loss = F.cross_entropy(output, y)
                atk_loss += loss.item() * y.size(0)
                atk_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
            torch.cuda.empty_cache()
        acc = 100*atk_acc/n
    else:
        holdout_net_2 = vgg19_bn().to(device)
        file_path = '../Standard-train-0322-1131'
        file_2 = "Train_Net_4/net_4.pth"
        holdout_net_2.load_state_dict(torch.load(os.path.join(file_path, file_2)))
        fgsm_attack_2 = torchattacks.FGSM(holdout_net_2, eps=0.007)
        fgsm_attack_2.save(data_loader=test_queue, file_name="../../cifar-data/tr_cifar10_fgsm_2.pt")
        fgsm_im2, fgsm_lab2 = torch.load("../../cifar-data/tr_cifar10_fgsm_2.pt")
        adv_data_3 = TensorDataset(fgsm_im2.float(), fgsm_lab2)
        adv_loader_fgsm = DataLoader(adv_data_3, batch_size=128, shuffle=False)
        acc = black_vgg_fgsm(adv_loader_fgsm, model)
        acc = 100*acc
        torch.cuda.empty_cache()
    return acc, attacks_name[type]

# Training
def train(train_queue, net, criterion, optimizer, params, epochs, save_pth, drop_path_prob):
    net.train()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))
    for epoch in range(epochs):

        net.droprate = drop_path_prob * epoch / epochs
        ls = 0
        correct = 0
        total = 0

        for step, (inputs, targets) in enumerate(train_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            outputs= net(inputs) # 训练标准网络用
            # outputs, outputs_aux = net(inputs) #搜索架构用
            loss = criterion(outputs, targets)

            # if params['auxiliary']:
            #     loss_aux = criterion(outputs_aux, targets)
            #     loss += params['auxiliary_weight'] * loss_aux
            net.zero_grad() #佳改
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), params['grad_clip'])
            optimizer.step()
            torch.cuda.empty_cache()

            ls += loss.data
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            torch.cuda.empty_cache()
        scheduler.step()
        # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        tr = 100. * correct / total

        logging.info('epoch %d train_acc %f', epoch, tr)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
        if epoch == 0:
            fp = open(os.path.join(save_pth, 'train.txt'), 'w')
        else:
            fp = open(os.path.join(save_pth, 'train.txt'), 'a+')
        fp.write('%f\n' % tr)
    print(tr)
    return tr, net


def ad_tr(train_queue, model, criterion, epochs, scheduler, delta_init, epsilon, pgd_alpha, alpha, opt, early_stop, out_dir, save_pth):

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_queue):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)
            if delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            if delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            output = model(X + delta[:X.size(0)])
            # output, outputs_aux = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            output = model(X + delta[:X.size(0)])
            # output, outputs_aux = model(X + delta[:X.size(0)])
            loss = criterion(output, y)
            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
            torch.cuda.empty_cache()
        if early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
                # output, _ = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        acc = 100*train_acc/train_n
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, acc)
    tr = acc
    train_time = time.time()
    if not early_stop:
        best_state_dict = model.state_dict()
    # torch.save(best_state_dict, os.path.join(save_pth, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)

    return tr, model


def infer(valid_queue, net, criterion):
    net.eval()

    valid_loss = 0
    correct = 0
    total = 0
    # tot = 0
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            # outputs, _ = net(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            torch.cuda.empty_cache()
            # if step % args.report_freq == 0:
            #     logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)
    acc = 100.*correct/total
    return acc


def new_infer(test_loader, net):
    model_test = net.cuda()
    # model_test.load_state_dict(best_state_dict)
    # file = "model_{}.pth".format(ind)
    # model_test.load_state_dict(torch.load(os.path.join(args.out_dir, file)))
    # model_test.float()
    model_test.eval()

    # pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    return test_acc


def black_vgg_fgsm(adv_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(adv_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_acc/n

if __name__ == "__main__":
    DARTS_V2 = [[[[3, 0], [3, 1]], [[3, 0], [3, 1]], [[3, 1], [2, 0]], [[2, 0], [5, 2]]],
               [[[0, 0], [0, 1]], [[2, 2], [0, 1]], [[0, 0], [2, 2]], [[2, 2], [0, 1]]]]
    start = time.time()
    print(main(genome=DARTS_V2, epochs=1, save='DARTS_V2_16', seed=1, init_channels=16,
               auxiliary=False, cutout=False, drop_path_prob=0.0))
    print('Time elapsed = {} mins'.format((time.time() - start)/60))
    # start = time.time()
    # print(main(genome=DARTS_V2, epochs=20, save='DARTS_V2_32', seed=1, init_channels=32))
    # print('Time elapsed = {} mins'.format((time.time() - start) / 60))

