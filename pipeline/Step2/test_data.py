import os
import sys
import argparse

import cv2
from PIL import Image
import numpy as np
import paddle
import torch
from reprod_log import ReprodLogger, ReprodDiffHelper

# from FM_paddle import train

# labeled_dataset_paddle, unlabeled_dataset_paddle, test_dataset_paddle, \
# labeled_trainloader_paddle, unlabeled_trainloader_paddle, test_loader_paddle = train.build_dataset_loader()

# labeled_dataset_torch, unlabeled_dataset_torch, test_dataset_torch, \
# labeled_trainloader_torch, unlabeled_trainloader_torch, test_loader_torch = train.build_dataset_loader()

parser = argparse.ArgumentParser(description='PyTorch Paddle FixMatch Training')
parser.add_argument('--mu', default=7, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of workers')
parser.add_argument('--num-labeled', type=int, default=4000,
                    help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true",
                    help="expand labels to fit eval steps")
parser.add_argument('--eval-step', default=1024, type=int,
                    help='number of eval steps to run')
parser.add_argument('--num-classes', default=10, type=int,
                    help='number of classes of dataset')
parser.add_argument('--batch-size', default=64, type=int,
                    help='train batchsize')
args = parser.parse_args()

np.random.seed(42)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    # np.random.shuffle(labeled_idx)
    labeled_idx.sort()
    # print(f"np random label index: {labeled_idx}")
    return np.arange(0, 4000), unlabeled_idx


def build_paddle_transform():
    from paddle.vision import transforms
    paddle_transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    return paddle_transform


def build_torch_transform():
    from torchvision import transforms
    torch_transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    return torch_transform


def build_paddle_data_pipeline(data_file=r"D:\code_sources\from_github\paddlepaddle\14s\FixMatch-Paddle\data\cifar-10-python.tar.gz"):
    from paddle.io import DataLoader, SequenceSampler, BatchSampler
    from paddle.vision import datasets
    sys.path.insert(0, "./FM_paddle")
    from FM_paddle.dataset.cifar import CIFAR10SSL

    transform_val = build_paddle_transform()
    base_dataset = datasets.Cifar10(data_file=data_file, mode='train', download=True, transform=transform_val)
    print(f"base_train_dataset paddle: {np.array(base_dataset[0][0])}")
    # print(base_dataset.data)
    # print(base_dataset.data.shape)  # (50000, 2)
    # for i in range(5):
    #     image, label = base_dataset[i]
    #     print(image.shape, label)
    base_sampler = SequenceSampler(base_dataset)
    base_batch_sampler = BatchSampler(sampler=base_sampler,
                                      batch_size=args.batch_size,
                                      drop_last=True)
    base_dataloader = DataLoader(dataset=base_dataset,
                                 batch_size=args.batch_size,
                                 drop_last=True,
                                 # batch_sampler=base_batch_sampler,
                                 num_workers=args.num_workers)
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, np.asarray(base_dataset.data)[:, 1])  # 取标签列

    train_labeled_dataset = CIFAR10SSL(data_file, train_labeled_idxs, mode='train', transform=transform_val)

    train_unlabeled_dataset = CIFAR10SSL(data_file, train_unlabeled_idxs, mode='train', transform=transform_val)

    test_dataset = datasets.Cifar10(data_file=data_file, mode='test', transform=transform_val, download=True)

    labeled_sampler = SequenceSampler(train_labeled_dataset)
    labeled_batch_sampler = BatchSampler(sampler=labeled_sampler,
                                         batch_size=args.batch_size,
                                         drop_last=True)
    labeled_trainloader = DataLoader(train_labeled_dataset,
                                     # batch_sampler=labeled_batch_sampler,
                                     num_workers=args.num_workers,)

    unlabeled_sampler = SequenceSampler(train_unlabeled_dataset)
    unlabeled_batch_sampler = BatchSampler(sampler=unlabeled_sampler,
                                           batch_size=args.batch_size*args.mu,
                                           drop_last=True)
    unlabeled_trainloader = DataLoader(train_unlabeled_dataset,
                                       # batch_sampler=unlabeled_batch_sampler,
                                       num_workers=args.num_workers,)

    test_sampler = SequenceSampler(test_dataset)
    test_batch_sampler = BatchSampler(sampler=test_sampler, batch_size=args.batch_size)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=args.num_workers)
    sys.path.pop(0)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, \
           labeled_trainloader, unlabeled_trainloader, test_loader, base_dataloader, base_dataset


def build_torch_data_pipeline(root=r"D:\code_sources\from_github\paddlepaddle\14s\FixMatch-Paddle\data"):
    from torchvision import datasets
    from torchvision import transforms
    from torch.utils.data import DataLoader, SequentialSampler
    sys.path.insert(0, "./FM_torch")
    from FM_torch.dataset.cifar import CIFAR10SSL

    transform_val = build_torch_transform()

    base_dataset = datasets.CIFAR10(root, train=True, download=True, transform=transform_val)
    print(f"base_train_dataset torch : {np.array(base_dataset[0][0])}")
    base_trainloader = DataLoader(
        base_dataset,
        # sampler=SequentialSampler(base_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    labeled_dataset = CIFAR10SSL(root, train_labeled_idxs, train=True, transform=transform_val)

    unlabeled_dataset = CIFAR10SSL(root, train_unlabeled_idxs, train=True, transform=transform_val)

    test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=False)

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=SequentialSampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=SequentialSampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)


    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    sys.path.pop(0)
    return labeled_dataset, unlabeled_dataset, test_dataset, \
           labeled_trainloader, unlabeled_trainloader, test_loader, base_trainloader, base_dataset


def test_transform():
    paddle_transform = build_paddle_transform()
    torch_transform = build_torch_transform()
    img = Image.open("./demo_image/ILSVRC2012_val_00006697.JPEG")

    paddle_img = paddle_transform(img)
    torch_img = torch_transform(img)

    np.testing.assert_allclose(paddle_img, torch_img)


def test_data_pipeline():
    diff_helper = ReprodDiffHelper()

    labeled_dataset_paddle, unlabeled_dataset_paddle, test_dataset_paddle, \
    labeled_trainloader_paddle, unlabeled_trainloader_paddle, test_loader_paddle, \
    base_trainloader_paddle, base_dataset_paddle = build_paddle_data_pipeline()

    labeled_dataset_torch, unlabeled_dataset_torch, test_dataset_torch, \
    labeled_trainloader_torch, unlabeled_trainloader_torch, test_loader_torch, \
    base_trainloader_torch, base_dataset_torch = build_torch_data_pipeline()



    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    logger_paddle_data.add("labeled_length", np.array(len(labeled_dataset_paddle)))
    logger_torch_data.add("labeled_length", np.array(len(labeled_dataset_torch)))

    logger_paddle_data.add("unlabeled_length", np.array(len(unlabeled_dataset_paddle)))
    logger_torch_data.add("unlabeled_length", np.array(len(unlabeled_dataset_torch)))

    logger_paddle_data.add("train_length", np.array(len(base_dataset_paddle)))
    logger_torch_data.add("train_length", np.array(len(base_dataset_torch)))

    logger_paddle_data.add("test_length", np.array(len(test_dataset_paddle)))
    logger_torch_data.add("test_length", np.array(len(test_dataset_torch)))

    # random choose 5 images and check
    for idx in range(5):
        labeled_rnd_idx = np.random.randint(0, len(labeled_dataset_paddle))
        unlabeled_rnd_idx = np.random.randint(0, len(unlabeled_dataset_paddle))
        test_rnd_idx = np.random.randint(0, len(test_dataset_paddle))

        # logger_paddle_data.add(f"labeled_dataset_{idx}",
        #                        labeled_dataset_paddle[labeled_rnd_idx][0].numpy())
        # print(f"labeled_dataset_padddle{idx}", labeled_dataset_paddle[labeled_rnd_idx][0].shape)
        # print(labeled_dataset_paddle[labeled_rnd_idx][1], labeled_dataset_torch[labeled_rnd_idx][1])
        # logger_torch_data.add(f"labeled_dataset_{idx}",
        #                       labeled_dataset_torch[labeled_rnd_idx][0].detach().cpu().numpy())
        # print(f"labeled_dataset_torch{idx}", labeled_dataset_torch[labeled_rnd_idx][0].detach().cpu().numpy().shape)
        #
        # logger_paddle_data.add(f"unlabeled_dataset_{idx}",
        #                        unlabeled_dataset_paddle[unlabeled_rnd_idx][0].numpy())
        # print(f"unlabeled_dataset_paddle{idx}", unlabeled_dataset_paddle[unlabeled_rnd_idx][0].numpy().shape)
        # logger_torch_data.add(f"unlabeled_dataset_{idx}",
        #                       unlabeled_dataset_torch[unlabeled_rnd_idx][0].numpy())
        # print(f"unlabeled_dataset_torch{idx}", unlabeled_dataset_torch[unlabeled_rnd_idx][0].detach().cpu().numpy().shape)
        logger_paddle_data.add(f"train_dataset_{idx}",
                               base_dataset_paddle[idx][0].numpy())
        print(f"unlabeled_dataset_paddle{idx}", unlabeled_dataset_paddle[unlabeled_rnd_idx][1])
        logger_torch_data.add(f"train_dataset_{idx}",
                              base_dataset_torch[idx][0].numpy())
        print(f"unlabeled_dataset_torch{idx}", unlabeled_dataset_torch[unlabeled_rnd_idx][1])

        logger_paddle_data.add(f"test_dataset_{idx}",
                               test_dataset_paddle[test_rnd_idx][0].numpy())
        logger_torch_data.add(f"test_dataset_{idx}",
                              test_dataset_torch[test_rnd_idx][0].detach().cpu().numpy())

    # for idx, (paddle_batch, torch_batch) in enumerate(zip(labeled_trainloader_paddle, labeled_trainloader_torch)):
    #     if idx >= 5:
    #         break
    #     logger_paddle_data.add(f"labeled_dataloader_{idx}", paddle_batch[0].numpy())
    #     logger_torch_data.add(f"labeled_dataloader_{idx}", torch_batch[0].detach().cpu().numpy())
    # for idx, (paddle_batch, torch_batch) in enumerate(zip(unlabeled_trainloader_paddle, unlabeled_trainloader_torch)):
    #     if idx >= 5:
    #         break
    #     logger_paddle_data.add(f"unlabeled_dataloader_{idx}", paddle_batch[0].numpy())
    #     logger_torch_data.add(f"unlabeled_dataloader_{idx}", torch_batch[0].detach().cpu().numpy())
    for idx, (paddle_batch, torch_batch) in enumerate(zip(base_trainloader_paddle, base_trainloader_torch)):
        if idx >= 5:
            break
        logger_paddle_data.add(f"train_dataloader_{idx}", paddle_batch[0].numpy())
        logger_torch_data.add(f"train_dataloader_{idx}", torch_batch[0].detach().cpu().numpy())
    for idx, (paddle_batch, torch_batch) in enumerate(zip(test_loader_paddle, test_loader_torch)):
        if idx >= 5:
            break
        logger_paddle_data.add(f"test_dataloader_{idx}", paddle_batch[0].numpy())
        logger_torch_data.add(f"test_dataloader_{idx}", torch_batch[0].detach().cpu().numpy())

    diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
    diff_helper.report()


if __name__ == "__main__":
    test_data_pipeline()
