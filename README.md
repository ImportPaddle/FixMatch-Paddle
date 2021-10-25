# FixMatch

使用 paddle 复现论文 [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685).
官方开源 pytorch 代码 [在这](https://github.com/google-research/fixmatch).

## 模型对齐

详见 `FixMatch-Paddle/pipeline/`

- Step1 模型结构对齐
- Step2 评估指标对齐
- Step3 损失函数和反向传播对齐
- Step4 训练对齐

## 复现精度

模型和训练日志在 Release 中，也可到百度网盘提取，链接：https://pan.baidu.com/s/14j8AQnvLgBFgCfZzHbfpxQ 
提取码：kimn

### CIFAR10

| #Labels | 40 | 250 | 4000 |
|:---:|:---:|:---:|:---:|
| Paper (RA) | 86.19 ± 3.37 | 94.93 ± 0.65 | 95.74 ± 0.05 |
| pytorch code | 93.60 | 95.22 | 95.77 |
| **paddle code** | 93.59 (epoch 190) | 95.20 (epoch 255) | 95.30 (epoch 156) |
| **model_best** | [model_best@40.pdparams](https://github.com/ImportPaddle/FixMatch-Paddle/releases/tag/trainv0.8) | [model_best@250.pdparams](https://github.com/ImportPaddle/FixMatch-Paddle/releases/tag/trainv0.8) | [model_best@4000.pdparams](https://github.com/ImportPaddle/FixMatch-Paddle/releases/tag/trainv0.8) |

\* paddle 精度截至 10.25 17:26。还在继续训练（单机4卡），250 labeled 和 4000 labeled 模型的精度还在提升

## 快速开始

cifar10 数据集: [cifar-10-python.tar.gz](https://github.com/ImportPaddle/FixMatch-Paddle/releases/tag/trainv0.8)

### Train

- **单卡：**
    - 命令行1：使用 4000（40、250） 个有标签样本，从零开始训练:
    
    ```
    python train.py --dataset cifar10 \
        --num-labeled 4000 \
        --arch wideresnet --batch-size 64 --lr 0.03 \
        --expand-labels --seed 5 \
        --log-out results/cifar10@4000.5/logs \
        --out results/cifar10@4000.5 \
        --data-file path/to/cifar10
    ```
    
    - 命令行2：从预训练模型继续训练，添加命令行参数 `--resume path/to/latest-ckpt`

- **单机 4 卡：**

    ```
    python -m paddle.distributed.launch --gpus '0,1,2,3' train.py \
        --dataset cifar10 \
        --num-labeled 4000 \
        --arch wideresnet --batch-size 64 --lr 0.095 \
        --expand-labels --seed 5 \
        --local_rank 0 \
        --log-out results/cifar10@4000.5/logs \
        --out results/cifar10@4000.5 \
        --data-file path/to/cifar10 \
        --resume path/to/latest-ckpt
    ```

### Test

- 方式1: 加载模型中保存的 dict:
```python
import paddle 
state=paddle.load("path/to/moddel_best.pdparams")
print(state.keys())  # 查看model中保存的dict
print('epoch:', state['epoch'])  # 查看是哪个epoch保存的最佳模型
print(f"best acc: {state['best_acc']}")  # 查看模型的测试集精度
```

- 方式2: 使用命令行
```
python test.py --dataset cifar10 \
    --arch wideresnet --batch-size 64 \
    --expand-labels --seed 5 \
    --data-file path/to/cifar10 \
    --model-best path/to/best-ckpt
```

输出如下：

```
Test Iter:    1/  22. Data: 0.895s. Batch: 0.942s. Loss: 0.2222. top1: 93.97. top5: 99.78. 
Test Iter:    2/  22. Data: 0.448s. Batch: 0.485s. Loss: 0.1925. top1: 94.75. top5: 99.67. 
Test Iter:    3/  22. Data: 0.299s. Batch: 0.332s. Loss: 0.2224. top1: 94.64. top5: 99.70. 
Test Iter:    4/  22. Data: 0.224s. Batch: 0.260s. Loss: 0.2255. top1: 94.70. top5: 99.78. 
Test Iter:    5/  22. Data: 0.183s. Batch: 0.216s. Loss: 0.2200. top1: 94.69. top5: 99.82. 
Test Iter:    6/  22. Data: 0.173s. Batch: 0.207s. Loss: 0.2203. top1: 94.79. top5: 99.81. 
Test Iter:    7/  22. Data: 0.148s. Batch: 0.181s. Loss: 0.2156. top1: 94.80. top5: 99.81. 
Test Iter:    8/  22. Data: 0.130s. Batch: 0.161s. Loss: 0.2135. top1: 94.81. top5: 99.75. 
Test Iter:    9/  22. Data: 0.123s. Batch: 0.154s. Loss: 0.2136. top1: 94.79. top5: 99.78. 
Test Iter:   10/  22. Data: 0.123s. Batch: 0.153s. Loss: 0.2218. top1: 94.73. top5: 99.75. 
Test Iter:   11/  22. Data: 0.111s. Batch: 0.142s. Loss: 0.2158. top1: 94.89. top5: 99.78. 
Test Iter:   12/  22. Data: 0.102s. Batch: 0.132s. Loss: 0.2067. top1: 95.07. top5: 99.80. 
Test Iter:   13/  22. Data: 0.100s. Batch: 0.129s. Loss: 0.2059. top1: 95.09. top5: 99.81. 
Test Iter:   14/  22. Data: 0.101s. Batch: 0.130s. Loss: 0.2047. top1: 95.03. top5: 99.81. 
Test Iter:   15/  22. Data: 0.094s. Batch: 0.124s. Loss: 0.2059. top1: 94.93. top5: 99.81. 
Test Iter:   16/  22. Data: 0.088s. Batch: 0.118s. Loss: 0.2019. top1: 95.03. top5: 99.82. 
Test Iter:   17/  22. Data: 0.087s. Batch: 0.116s. Loss: 0.2062. top1: 94.96. top5: 99.83. 
Test Iter:   18/  22. Data: 0.089s. Batch: 0.117s. Loss: 0.2100. top1: 94.90. top5: 99.84. 
Test Iter:   19/  22. Data: 0.084s. Batch: 0.112s. Loss: 0.2080. top1: 94.94. top5: 99.84. 
Test Iter:   20/  22. Data: 0.080s. Batch: 0.108s. Loss: 0.2073. top1: 94.96. top5: 99.84. 
Test Iter:   21/  22. Data: 0.078s. Batch: 0.106s. Loss: 0.2058. top1: 95.00. top5: 99.85. 
Test Iter:   22/  22. Data: 0.080s. Batch: 0.108s. Loss: 0.2073. top1: 94.99. top5: 99.85. 
top-1 acc: 94.99
top-5 acc: 99.85
```

## 环境依赖

- python 3.7
- paddle 2.1
- visualdl
- numpy
- tqdm

## Citations

```
@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}
```
