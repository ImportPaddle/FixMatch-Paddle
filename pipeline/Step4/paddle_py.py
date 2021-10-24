from pipeline.utils import *
from pipeline.Step2.Evaluate_paddle import accuracy as accuracy_paddle
from pipeline.Step2.Evaluate_torch import accuracy as accuracy_torch
from pipeline.Step2.Evaluate_paddle import AverageMeter as AverageMeter_paddle
from pipeline.Step2.Evaluate_paddle import AverageMeter as AverageMeter_torch
import paddle
import torch




def test(args, test_loader, model, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with paddle.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.eval()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])


        if not args.no_progress:
            test_loader.close()

    return losses.avg, top1.avg