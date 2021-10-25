# FixMatch-Paddle


FixMatch-Paddle的Step1主要进行的是模型的结构对齐。FixMatch-Paddle包含两个网络，renext和wideresnet，模型对齐时将分别将两个网络进行对齐。

注意：运行前要将forward_wideresnet.py中的文件地址进行修改！
```shell
#进入文件夹
cd pipeline/Step1/
#分别生成网络的前向数据，每个数据都分别包括torch数据和paddle数据
python3.7 forward_resnext.py
python3.7 forward_wideresnet.py
```

forward_resnext的代码如下，代码将包括diff检查代码

````python
import numpy as np

from pipeline.Step1.models import resnext_paddle,resnext_torch
from reprod_log import ReprodDiffHelper,ReprodLogger
import torch
import paddle

model_name='resnext'

def model_set():
    resnext_paddle_b=resnext_paddle.build_resnext(cardinality=4,depth=28,width=4,num_classes=10)
    resnext_torch_b=resnext_torch.build_resnext(cardinality=4,depth=28,width=4,num_classes=10)
    return resnext_paddle_b,resnext_torch_b


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
        if 'classifier.weight' == key:
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
    
    #diff检查
    reprod_log_paddle.save(f"./{model_name}_paddle.npy")
    reprod_log_torch.save(f"./{model_name}_torch.npy")

def gen_check():
    diff_helper = ReprodDiffHelper()
    info_torch = diff_helper.load_info(f"./{model_name}_torch.npy")
    info_paddle = diff_helper.load_info(f"./{model_name}_paddle.npy")

    diff_helper.compare_info(info_paddle,info_torch)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path=f"./diff_{model_name}_model.txt")

````

产出日志如下

```resnext
[2021/10/24 11:43:24] root INFO: data_1: 
[2021/10/24 11:43:24] root INFO: 	mean diff: check passed: True, value: 5.397714719634905e-09
[2021/10/24 11:43:24] root INFO: data_2: 
[2021/10/24 11:43:24] root INFO: 	mean diff: check passed: True, value: 5.359316990194429e-09
[2021/10/24 11:43:24] root INFO: data_3: 
[2021/10/24 11:43:24] root INFO: 	mean diff: check passed: True, value: 5.321862950324885e-09
[2021/10/24 11:43:24] root INFO: diff check passed
```

平均绝对误差为5.3e-9，测试通过。

forward_wideresnet的代码如下，代码将包括diff检查代码

````python
import numpy as np

from pipeline.Step1.models import wideresnet_torch, wideresnet_paddle
from reprod_log import ReprodDiffHelper, ReprodLogger
import torch
import paddle

model_name = 'wideresnet'


def model_set():
    wideresnet_paddle_b = wideresnet_paddle.build_wideresnet(depth=28, widen_factor=2, dropout=0, num_classes=10)
    wideresnet_torch_b = wideresnet_torch.build_wideresnet(depth=28, widen_factor=2, dropout=0, num_classes=10)
    return wideresnet_paddle_b, wideresnet_torch_b


def data_set(seed):
    batch_size = 64
    channel = 3
    input_w = 32
    input_H = 32
    np.random.seed(seed)
    data = np.random.randn(batch_size, channel, input_w, input_H).astype(np.float32)
    data_paddle, data_torch = paddle.to_tensor(data), torch.from_numpy(data)
    return data_paddle, data_torch


def gen_params(model_torch):
    model_torch_params = model_torch.state_dict()
    model_paddle_params = {}
    for key in model_torch_params:
        weight = model_torch_params[key].cpu().detach().numpy()
        if 'running_mean' in key:
            key = key.replace('running_mean', '_mean')
        if 'running_var' in key:
            key = key.replace('running_var', '_variance')
        if 'fc.weight' == key:
            weight = weight.transpose()
        if 'classifier.weight' == key:
            weight = weight.transpose()
        model_paddle_params[key] = weight
    return model_paddle_params, model_torch_params

def gen_npy(seed_list):
    reprod_log_paddle = ReprodLogger()
    reprod_log_torch = ReprodLogger()
    model_paddle, model_torch=model_set()
    # 更换一下绝对路径
    state=torch.load('/Users/yangruizhi/Desktop/PR_list/FixMatch-Paddle/pipeline/Step1/models/model_best.pth.tar')
    model_torch.load_state_dict(state['state_dict'])
    model_paddle.eval()
    model_torch.eval()
    for seed in seed_list:
        data_paddle, data_torch = data_set(seed)
        params_paddle, params_torch = gen_params(model_torch)
        model_paddle.set_state_dict(params_paddle)
        model_torch.load_state_dict(params_torch)
        res_paddle, res_torch = model_paddle(data_paddle), model_torch(data_torch)
        reprod_log_paddle.add(f"data_{seed_list.index(seed) + 1}", res_paddle.numpy())
        reprod_log_torch.add(f"data_{seed_list.index(seed) + 1}", res_torch.data.cpu().numpy())
    
    #diff检查
    reprod_log_paddle.save(f"./{model_name}_paddle.npy")
    reprod_log_torch.save(f"./{model_name}_torch.npy")


def gen_check():
    diff_helper = ReprodDiffHelper()
    info_torch = diff_helper.load_info(f"./{model_name}_torch.npy")
    info_paddle = diff_helper.load_info(f"./{model_name}_paddle.npy")

    diff_helper.compare_info(info_paddle, info_torch)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path=f"./diff_{model_name}_model.txt")
````
产出日志如下
```
[2021/10/24 11:52:03] root INFO: data_1: 
[2021/10/24 11:52:03] root INFO: 	mean diff: check passed: True, value: 4.1866078959174047e-07
[2021/10/24 11:52:03] root INFO: data_2: 
[2021/10/24 11:52:03] root INFO: 	mean diff: check passed: True, value: 4.308720349399664e-07
[2021/10/24 11:52:03] root INFO: data_3: 
[2021/10/24 11:52:03] root INFO: 	mean diff: check passed: True, value: 4.4478801441982796e-07
[2021/10/24 11:52:03] root INFO: diff check passed

```

平均绝对误差为4.2e-7，测试通过。