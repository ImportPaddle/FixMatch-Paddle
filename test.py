import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
import numpy as np

import paddle
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.optimizer.lr import LambdaDecay
from paddle.io import DataLoader, RandomSampler, SequenceSampler, BatchSampler, DistributedBatchSampler
import paddle.distributed as dist
# from visualdl import LogWriter

from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

logger = logging.getLogger()
# logger_2 = logging.getLogger()

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
    return x.reshape([-1, size] + s[1:]).transpose([1, 0, 2, 3, 4]).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose([1, 0, 2]).reshape([-1] + s[1:])


def main():
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
    parser.add_argument('--data-file', default='./data/cifar-10-python.tar.gz', type=str,
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
    # args.writer = LogWriter(logdir=args.out)
    os.makedirs(args.out, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        filename=f'{args.out}/train@{args.num_labeled}.log',
        filemode='a'
    )

    # BASIC_FORMAT = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    # DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

    # formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    logger.addHandler(chlr)
    # chlr.setFormatter(formatter)
    # chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
    # fhlr = logging.FileHandler('example.log') # 输出到文件的handler
    # fhlr.setFormatter(formatter)
    # logger.addHandler(fhlr)
    # logger.info('==============================logger success !')
    # raise NotImplemented

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", )
    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

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

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, args.data_file)

    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

        labeled_batch_sampler = DistributedBatchSampler(labeled_dataset,
                                                        batch_size=args.batch_size,
                                                        drop_last=True, shuffle=True)
        labeled_trainloader = DataLoader(
            labeled_dataset,
            batch_sampler=labeled_batch_sampler,
            num_workers=args.num_workers)

        labeled_batch_sampler = DistributedBatchSampler(unlabeled_dataset,
                                                        batch_size=args.batch_size * args.mu,
                                                        drop_last=True, shuffle=True)
        unlabeled_trainloader = DataLoader(
            unlabeled_dataset,
            batch_sampler=labeled_batch_sampler,
            num_workers=args.num_workers)

        test_sampler = RandomSampler(test_dataset)
        labeled_batch_sampler = BatchSampler(sampler=test_sampler,
                                             batch_size=args.batch_size * args.mu,
                                             drop_last=False)
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=labeled_batch_sampler,
            num_workers=args.num_workers)
    else:
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
                                             drop_last=False)
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=labeled_batch_sampler,
            num_workers=args.num_workers)

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

        checkpoint = paddle.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.set_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.set_state_dict(checkpoint['ema_state_dict'])
        optimizer_1.set_state_dict(checkpoint['optimizer_1'])
        optimizer_2.set_state_dict(checkpoint['optimizer_2'])
        if 'FixMatch-Paddle/params' in args.resume:
            args.resume = args.out
            args.out = os.path.dirname(args.resume)

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
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
    logger.info('===============stage one================')
    test(args, test_loader, model)

def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    model.eval()

    with paddle.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                if dist.get_rank() == 0:
                    logger.info(
                        "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                            batch=batch_idx + 1,
                            iter=len(test_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        ))
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    model.train()
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
