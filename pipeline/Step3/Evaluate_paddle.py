'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging

# import torch
import paddle

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = paddle.io.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = paddle.zeros(3)
    std = paddle.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target=paddle.expand_as(target.reshape([1, -1]),pred)
    correct = pred.equal(target)

    res = []
    for k in topk:
        # print("correct_k:",correct.astype(paddle.int64)[:k].sum(1))
        # correct_k = correct[:k].reshape([-1]).astype(paddle.float32).sum(0)
        correct_k=correct.astype(paddle.float32)[:k].sum(1).sum()
        res.append(correct_k.multiply(paddle.to_tensor(100.0 / batch_size)))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count