import numpy as np

from models import wideresnet_torch,wideresnet_paddle
from reprod_log import ReprodDiffHelper,ReprodLogger
import torch
import paddle

model_name='wideresnet'

def model_set():
    wideresnet_paddle_b=wideresnet_paddle.build_wideresnet(depth=28,widen_factor=2,dropout=0,num_classes=10)
    wideresnet_torch_b=wideresnet_torch.build_wideresnet(depth=28,widen_factor=2,dropout=0,num_classes=10)
    return wideresnet_paddle_b,wideresnet_torch_b


def data_set(seed):
    batch_size=64
    channel=3
    input_w=32
    input_H=32
    np.random.seed(seed)
    data=np.random.randn(batch_size,channel,input_w,input_H).astype(np.float32)
    data_paddle,data_torch=paddle.to_tensor(data),torch.from_numpy(data)
    return data_paddle,data_torch

def gen_params(model_1):
    model_1_params=model_1.state_dict()
    model_2_params = {}
    for key in model_1_params:
        weight = model_1_params[key].cpu().detach().numpy()
        if 'running_mean' in key :
            key=key.replace('running_mean','_mean')
        if 'running_var' in key:
            key=key.replace('running_var','_variance')
        if 'fc.weight' == key:
            weight=weight.transpose()
        model_2_params[key]=weight
    return model_2_params,model_1_params

def gen_npy(seed_list):
    reprod_log_paddle = ReprodLogger()
    reprod_log_torch = ReprodLogger()
    model_paddle, model_torch = model_set()
    model_paddle.eval()
    model_torch.eval()
    for seed in seed_list:
        data_paddle,data_torch=data_set(seed)
        params_paddle,params_torch=gen_params(model_torch)
        model_paddle.set_state_dict(params_paddle)
        model_torch.load_state_dict(params_torch)
        res_paddle,res_torch=model_paddle(data_paddle),model_torch(data_torch)
        reprod_log_paddle.add(f"data_{seed_list.index(seed)+1}",res_paddle.numpy())
        reprod_log_torch.add(f"data_{seed_list.index(seed)+1}",res_torch.data.cpu().numpy())
    reprod_log_paddle.save(f"./{model_name}_paddle.npy")
    reprod_log_torch.save(f"./{model_name}_torch.npy")

def gen_check():
    diff_helper = ReprodDiffHelper()
    info_torch = diff_helper.load_info(f"./{model_name}_torch.npy")
    info_paddle = diff_helper.load_info(f"./{model_name}_paddle.npy")

    diff_helper.compare_info(info_paddle,info_torch)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path=f"./diff_{model_name}_model.txt")
if __name__ == '__main__':
    seed_list=[100,300,500]
    gen_npy(seed_list)
    gen_check()

