from pipeline.utils import *
from pipeline.Step3.Evaluate_paddle import accuracy as accuracy_paddle
from pipeline.Step3.Evaluate_torch import accuracy as accuracy_torch
from pipeline.Step3.Evaluate_paddle import AverageMeter as AverageMeter_paddle
from pipeline.Step3.Evaluate_paddle import AverageMeter as AverageMeter_torch
import paddle
import torch
import torch_py
import paddle_py

def interleave_paddle(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose([1,0,2,3,4]).reshape([-1] + s[1:])


def de_interleave_paddle(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose([1,0,2]).reshape([-1] + s[1:])

def interleave_torch(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave_torch(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])



if __name__ == '__main__':
    model_name = 'resnext'
    save_name='Train_Alignment'
    epoch_num=10
    model_paddle, model_torch = gen_model(model_name=model_name)
    model_paddle, model_torch = update_model(model_paddle, model_torch)

    top1_paddle, top5_paddle = AverageMeter_paddle(), AverageMeter_paddle()
    top1_torch, top5_torch = AverageMeter_torch(), AverageMeter_torch()
    reprod_log_paddle = ReprodLogger()
    reprod_log_torch = ReprodLogger()
    args=get_args()
    labeled_trainloader, unlabeled_trainloader, test_loader=gen_dataloader_paddle(args)

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    model_paddle.train()
    model_torch.train()
    for epoch in range(epoch_num):
        try:
            inputs_x, targets_x = labeled_iter.next()
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_trainloader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_iter.next()
        try:
            (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_trainloader)
            (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
        batch_size = inputs_x.shape[0]
        inputs_paddle = interleave_paddle(
            paddle.concat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1)
        inputs_torch=data_paddle_2_torch(inputs_paddle)

        logits_paddle, logits_torch = gen_res(model_paddle, inputs_paddle, model_torch, inputs_torch)

        #  计算loss_paddle
        logits_paddle = de_interleave_paddle(logits_paddle, 2 * args.mu + 1)
        logits_x_paddle = logits_paddle[:batch_size]
        logits_u_w_paddle, logits_u_s_paddle = logits_paddle[batch_size:].chunk(2)
        del logits_paddle

        Lx = F.cross_entropy(logits_x_paddle, targets_x, reduction='mean')

        pseudo_label = F.softmax(logits_u_w_paddle.detach() / args.T, axis=-1)

        max_probs, targets_u = paddle.max(pseudo_label, axis=-1), paddle.argmax(pseudo_label, axis=-1)
        mask = paddle.greater_equal(max_probs, paddle.to_tensor(args.threshold)).astype(paddle.float32)

        Lu = (F.cross_entropy(logits_u_s_paddle, targets_u,
                              reduction='none') * mask).mean()

        loss_paddle = Lx + args.lambda_u * Lu
        # loss_paddle 计算结束

        # 计算 loss_torch
        logits_torch = de_interleave_torch(logits_torch, 2 * args.mu + 1)
        logits_x_torch = logits_torch[:batch_size]
        logits_u_w_torch, logits_u_s_torch = logits_torch[batch_size:].chunk(2)

        del logits_torch

        Lx = torch.nn.functional.cross_entropy(logits_x_torch, data_paddle_2_torch(targets_x), reduction='mean')

        pseudo_label = torch.nn.functional.softmax(logits_u_w_torch.detach() / args.T, axis=-1)

        max_probs, targets_u = paddle.max(pseudo_label, axis=-1), paddle.argmax(pseudo_label, axis=-1)
        mask = paddle.greater_equal(max_probs, args.threshold).float()

        Lu = (torch.nn.functional.cross_entropy(logits_u_s_torch, targets_u,
                              reduction='none') * mask).mean()

        loss_torch = Lx + args.lambda_u * Lu
        # 计算 loss_torch 结束
        loss_paddle.backward()
        loss_torch.backward()

        test_res_paddle=paddle_py.test(args,test_loader,model_paddle)
        test_res_torch=torch_py.test(args,test_loader,model_torch)

        reprod_log_paddle.add(f"epoch_{epoch + 1}_top1", np.array(test_res_paddle))
        reprod_log_torch.add(f"epoch_{epoch + 1}_top5", np.array(test_res_torch))
    reprod_log_paddle.save(f"./{save_name}_paddle.npy")
    reprod_log_torch.save(f"./{save_name}_torch.npy")
    gen_check(save_name)

