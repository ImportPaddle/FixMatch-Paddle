import numpy as np

from pipeline.Step1.models import wideresnet_torch, wideresnet_paddle

from pipeline.Step1.models import resnext_paddle, resnext_torch
from reprod_log import ReprodDiffHelper, ReprodLogger
import torch

import paddle
import paddle.nn.functional as F
import paddle.optimizer as optim

from paddle.optimizer.lr import LambdaDecay
from paddle.io import DataLoader, RandomSampler, SequenceSampler, BatchSampler

from tqdm import tqdm
import argparse
import logging
import math
import os
import random
import shutil
import time

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy


def gen_model(model_name='resnext'):
    if model_name == 'resnext':
        resnext_paddle_b = resnext_paddle.build_resnext(cardinality=4, depth=28, width=4, num_classes=10)
        resnext_torch_b = resnext_torch.build_resnext(cardinality=4, depth=28, width=4, num_classes=10)
        return resnext_paddle_b, resnext_torch_b
    if model_name == 'wideresnet':
        wideresnet_paddle_b = wideresnet_paddle.build_wideresnet(depth=28, widen_factor=2, dropout=0, num_classes=10)
        wideresnet_torch_b = wideresnet_torch.build_wideresnet(depth=28, widen_factor=2, dropout=0, num_classes=10)
        return wideresnet_paddle_b, wideresnet_torch_b


def gen_fake_data(seed=100, shape=None):
    if shape is None:
        shape = [64, 3, 32, 32]
    batch_size, channel, input_w, input_H = shape
    np.random.seed(seed)
    data = np.random.randn(batch_size, channel, input_w, input_H).astype(np.float32)
    data_paddle, data_torch = paddle.to_tensor(data), torch.from_numpy(data)
    return data_paddle, data_torch
def data_paddle_2_torch(data_paddle):
    return torch.from_numpy(data_paddle.numpy())

def gen_fake_label(seed=100, shape=None, num_classes=10):
    if shape is None:
        shape = 64
    np.random.seed(seed)
    fake_label = np.random.randint(0, 10, shape)
    label_paddle, label_torch = paddle.to_tensor(fake_label), torch.from_numpy(fake_label)
    return label_paddle, label_torch


def gen_params(model_1):
    model_1_params = model_1.state_dict()
    model_2_params = {}
    for key in model_1_params:
        weight = model_1_params[key].cpu().detach().numpy()
        if 'running_mean' in key:
            key = key.replace('running_mean', '_mean')
        if 'running_var' in key:
            key = key.replace('running_var', '_variance')
        if 'classifier.weight' == key:
            weight = weight.transpose()
        model_2_params[key] = weight
    return model_2_params, model_1_params


def gen_res(model_paddle, data_paddle, model_torch, data_torch):
    return model_paddle(data_paddle), model_torch(data_torch)


def update_model(model_paddle, model_torch):
    params_paddle, params_torch = gen_params(model_torch)
    model_paddle.set_state_dict(params_paddle)
    model_torch.load_state_dict(params_torch)
    return model_paddle, model_torch


def gen_npy(seed_list, model_name='resnext'):
    reprod_log_paddle = ReprodLogger()
    reprod_log_torch = ReprodLogger()
    model_paddle, model_torch = gen_model(model_name)
    model_paddle.eval()
    model_torch.eval()
    for seed in seed_list:
        data_paddle, data_torch = gen_fake_data(seed)
        params_paddle, params_torch = gen_params(model_torch)
        model_paddle.set_state_dict(params_paddle)
        model_torch.load_state_dict(params_torch)
        res_paddle, res_torch = model_paddle(data_paddle), model_torch(data_torch)
        reprod_log_paddle.add(f"data_{seed_list.index(seed) + 1}", res_paddle.numpy())
        reprod_log_torch.add(f"data_{seed_list.index(seed) + 1}", res_torch.data.cpu().numpy())
    reprod_log_paddle.save(f"./{model_name}_paddle.npy")
    reprod_log_torch.save(f"./{model_name}_torch.npy")


def torch2paddle(params_torch):
    model_1_params = params_torch
    model_2_params = {}
    for key in model_1_params:
        weight = model_1_params[key].cpu().detach().numpy()
        if 'running_mean' in key:
            key = key.replace('running_mean', '_mean')
        if 'running_var' in key:
            key = key.replace('running_var', '_variance')
        if 'classifier.weight' == key:
            weight = weight.transpose()
        if 'fc.weight' == key:
            weight = weight.transpose()
        model_2_params[key] = weight
    return model_2_params


def torch2paddle_ema(params_torch):
    model_1_params = params_torch
    model_2_params = {}
    for key in model_1_params:
        weight = model_1_params[key].cpu().detach().numpy()
        if 'running_mean' in key:
            key = key.replace('running_mean', '_mean')
        if 'running_var' in key:
            key = key.replace('running_var', '_variance')
        if 'classifier.weight' == key:
            weight = weight.transpose()
        if 'fc.weight' == key:
            weight = weight.transpose()
        model_2_params[key] = weight
    return model_2_params


def gen_check(name):
    diff_helper = ReprodDiffHelper()
    info_torch = diff_helper.load_info(f"./{name}_torch.npy")
    info_paddle = diff_helper.load_info(f"./{name}_paddle.npy")

    diff_helper.compare_info(info_paddle, info_torch)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path=f"./diff_{name}_model.txt")

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pdparams'):
    filepath = os.path.join(checkpoint, filename)
    paddle.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pdparams'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def get_cosine_schedule_with_warmup(learning_rate, num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaDecay(learning_rate=learning_rate,
                       lr_lambda=_lr_lambda,
                       last_epoch=last_epoch,
                       # verbose=True,
                       )


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose([1,0,2,3,4]).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose([1,0,2]).reshape([-1] + s[1:])


def get_args():
    parser = argparse.ArgumentParser(description='Paddle FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2 ** 20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data-file', default='../../data/cifar-10-python.tar.gz', type=str,
                        help='path to cifar10 dataset')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()
    global best_acc

    paddle.set_device('gpu') if paddle.is_compiled_with_cuda() else paddle.set_device('cpu')
    args.n_gpu = len(paddle.static.cuda_places()) if paddle.is_compiled_with_cuda() else 0

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            (sum(p.numel() for p in model.parameters()) / 1e6).numpy()[0]))
        return model

    args.device = paddle.get_device()
    args.world_size = 1

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)
    #
    # if args.local_rank in [-1, 0]:
    #     os.makedirs(args.out, exist_ok=True)
    #     args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    model = create_model(args)

    no_decay = ['bias', 'bn']

    scheduler_1 = get_cosine_schedule_with_warmup(args.lr, args.warmup, args.total_steps)
    scheduler_2 = get_cosine_schedule_with_warmup(args.lr, args.warmup, args.total_steps)

    model_params_1 = [p for n, p in model.named_parameters() if not any(
        nd in n for nd in no_decay)]
    model_params_2 = [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)]

    optimizer_1 = optim.Momentum(learning_rate=scheduler_1, momentum=0.9, weight_decay=args.wdecay,
                                 parameters=model_params_1, use_nesterov=args.nesterov)
    optimizer_2 = optim.Momentum(learning_rate=scheduler_2, momentum=0.9, weight_decay=0.0,
                                 parameters=model_params_2, use_nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model)

    args.start_epoch = 0
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = paddle.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer_1.load_state_dict(checkpoint['optimizer_1'])
        optimizer_2.load_state_dict(checkpoint['optimizer_2'])

    if args.amp:
        from apex import amp
        model, optimizer_1 = amp.initialize(
            model, optimizer_1, opt_level=args.opt_level)
        model, optimizer_2 = amp.initialize(
            model, optimizer_2, opt_level=args.opt_level)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    optimizer_1.clear_grad()
    optimizer_2.clear_grad()

    return args


def model_2_ema(args, model):
    if args.use_ema:
        from models.ema import ModelEMA
        model_ema = ModelEMA(args, model)
        return model_ema
    else:
        raise EOFError('meile ')

def gen_dataloader_paddle(args):
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, args.data_file)

    labeled_sampler = RandomSampler(labeled_dataset)
    labeled_batch_sampler = BatchSampler(sampler=labeled_sampler,
                                         batch_size=args.batch_size,
                                         drop_last=True)
    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_sampler=labeled_batch_sampler,
        num_workers=args.num_workers)

    unlabeled_sampler = RandomSampler(unlabeled_dataset)
    labeled_batch_sampler = BatchSampler(sampler=unlabeled_sampler,
                                         batch_size=args.batch_size * args.mu,
                                         drop_last=True)
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_sampler=labeled_batch_sampler,
        num_workers=args.num_workers)

    test_sampler = RandomSampler(test_dataset)
    labeled_batch_sampler = BatchSampler(sampler=test_sampler,
                                         batch_size=args.batch_size * args.mu,
                                         drop_last=True)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=labeled_batch_sampler,
        num_workers=args.num_workers)
    return labeled_trainloader,unlabeled_trainloader,test_loader

if __name__ == '__main__':
    pre_params = torch.load('/Users/yangruizhi/Desktop/PR_list/FixMatch-Paddle/pipeline/utils/model_best.pth.tar')
    print(pre_params.keys())
    params_paddle={}
    model_name = 'wideresnet'
    model_paddle, model_torch = gen_model(model_name)
    for key in pre_params.keys()[:5]:
        if key=='state_dict':
            params_paddle[key] = torch2paddle_ema(pre_params[key])
            continue
        if key=='ema_state_dict':
            params_paddle[key]


    model_torch.ema.load_state_dict(params_torch)
    model_paddle.ema.set_state_dict(params_paddle)
    paddle.save(params_paddle, '../model_params/pre_params_ema_paddle.pdparams')
    print('success!!!')