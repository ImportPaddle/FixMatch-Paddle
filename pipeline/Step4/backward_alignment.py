from pipeline.Step4 import optimizer_torch
from pipeline.Step4 import optimizer_paddle
from pipeline.Step3.Evaluate_paddle import AverageMeter as AverageMeter_paddle
from pipeline.Step3.Evaluate_paddle import AverageMeter as AverageMeter_torch
from pipeline.utils import *
import paddle
import torch.nn.functional as F

if __name__ == '__main__':
    model_name = 'resnext'
    save_name = 'backward_Alignment'
    seed_list = [100, 300, 1000,2000,3000]
    model_paddle, model_torch = gen_model(model_name=model_name)
    model_paddle, model_torch = update_model(model_paddle, model_torch)
    model_paddle.eval()
    model_torch.eval()
    losses_paddle = AverageMeter_paddle()
    losses_torch = AverageMeter_torch()
    optimizer_paddle_1,optimizer_paddle_2=optimizer_paddle.gen_optimizer(model_paddle)
    optimizer_torch=optimizer_torch.gen_optimizer(model_torch)
    reprod_log_paddle = ReprodLogger()
    reprod_log_torch = ReprodLogger()
    for seed in seed_list:
        fake_data, fake_label = gen_fake_data(seed), gen_fake_label(seed)
        data_paddle, data_torch = fake_data
        label_paddle, label_torch = fake_label
        res_paddle, res_torch = gen_res(model_paddle, data_paddle, model_torch, data_torch)
        loss_paddle=paddle.nn.functional.cross_entropy(res_paddle,label_paddle,reduction='mean')
        loss_torch =F.cross_entropy(res_torch,label_torch, reduction='mean')
        loss_paddle.backward()
        loss_torch.backward()
        losses_paddle.update(loss_paddle.item())
        losses_torch.update(loss_torch.item())
        optimizer_paddle_1.step()
        optimizer_paddle_2.step()
        optimizer_torch.step()
        optimizer_paddle_1.clear_grad()
        optimizer_paddle_2.clear_grad()
        model_torch.zero_grad()
        reprod_log_paddle.add(f"data_{seed_list.index(seed) + 1}_loss", np.array(losses_paddle.val))
        reprod_log_torch.add(f"data_{seed_list.index(seed) + 1}_loss", np.array(losses_torch.val))
    reprod_log_paddle.save(f"./{save_name}_paddle.npy")
    reprod_log_torch.save(f"./{save_name}_torch.npy")
    gen_check(save_name)

