from pipeline.utils import *
from pipeline.Step3.Evaluate_paddle import accuracy as accuracy_paddle
from pipeline.Step3.Evaluate_torch import accuracy as accuracy_torch
from pipeline.Step3.Evaluate_paddle import AverageMeter as AverageMeter_paddle
from pipeline.Step3.Evaluate_paddle import AverageMeter as AverageMeter_torch
import paddle
import torch

if __name__ == '__main__':
    model_name = 'resnext'
    seed_list = [100, 300, 1000]
    model_paddle, model_torch = gen_model(model_name=model_name)
    model_paddle, model_torch = update_model(model_paddle, model_torch)
    model_paddle.eval()
    model_torch.eval()
    top1_paddle, top5_paddle = AverageMeter_paddle(), AverageMeter_paddle()
    top1_torch, top5_torch = AverageMeter_torch(), AverageMeter_torch()
    reprod_log_paddle = ReprodLogger()
    reprod_log_torch = ReprodLogger()
    for seed in seed_list:
        fake_data, fake_label = gen_fake_data(seed), gen_fake_label(seed)
        data_paddle, data_torch = fake_data
        label_paddle, label_torch = fake_label
        res_paddle, res_torch = gen_res(model_paddle, data_paddle, model_torch, data_torch)
        prec1_paddle, prec5_paddle = accuracy_paddle(res_paddle, label_paddle, topk=(1, 5))
        prec1_torch, prec5_torch = accuracy_torch(res_torch, label_torch, topk=(1, 5))
        top1_paddle.update(prec1_paddle.item(), data_paddle.shape[0])
        top5_paddle.update(prec5_paddle.item(), data_paddle.shape[0])
        top1_torch.update(prec1_torch.item(), data_torch.size(0))
        top5_torch.update(prec5_torch.item(), data_torch.size(0))
        reprod_log_paddle.add(f"data_{seed_list.index(seed) + 1}_top1", np.array(top1_paddle.avg))
        reprod_log_paddle.add(f"data_{seed_list.index(seed) + 1}_top5", np.array(top5_paddle.avg))
        reprod_log_torch.add(f"data_{seed_list.index(seed) + 1}_top1", np.array(top1_torch.avg))
        reprod_log_torch.add(f"data_{seed_list.index(seed) + 1}_top5", np.array(top5_torch.avg))
    gen_check(reprod_log_paddle, reprod_log_torch, "Evaluate_Alignment")
