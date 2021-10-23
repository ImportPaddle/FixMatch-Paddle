import numpy as np

from pipeline.Step1.models import wideresnet_torch, wideresnet_paddle

from pipeline.Step1.models import resnext_paddle, resnext_torch
from reprod_log import ReprodDiffHelper, ReprodLogger
import torch
import paddle


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


def gen_fake_label(seed=100, shape=None,num_classes=10):
    if shape is None:
        shape = 64
    np.random.seed(seed)
    fake_label = np.random.randint(0,10,shape)
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


def gen_check(name):
    diff_helper = ReprodDiffHelper()
    info_torch = diff_helper.load_info(f"./{name}_torch.npy")
    info_paddle = diff_helper.load_info(f"./{name}_paddle.npy")

    diff_helper.compare_info(info_paddle, info_torch)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path=f"./diff_{name}_model.txt")


