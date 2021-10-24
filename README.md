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

### CIFAR10

| #Labels | 40 | 250 | 4000 |
|:---:|:---:|:---:|:---:|
| Paper (RA) | 86.19 ± 3.37 | 94.93 ± 0.65 | 95.74 ± 0.05 |
| pytorch code | 93.60 | 95.31 | 95.77 |
| **paddle code** | 92.89 (epoch 84) | 93.08 (epoch 77) | 93.99 (epoch 85) |
| **model_best** | [model_best@40.pdparams](https://github.com/ImportPaddle/FixMatch-Paddle/releases/tag/trainv0.8) | [model_best@250.pdparams](https://github.com/ImportPaddle/FixMatch-Paddle/releases/tag/trainv0.8) | [model_best@4000.pdparams](https://github.com/ImportPaddle/FixMatch-Paddle/releases/tag/trainv0.8) |

\* paddle 精度截至 10.24 17:22。还在继续训练（单卡），精度还在**缓慢**提升

## 快速开始

cifar10 数据集: [cifar-10-python.tar.gz](https://github.com/ImportPaddle/FixMatch-Paddle/releases/tag/trainv0.8)

### Train

- 命令行1：使用 4000（40、250） 个有标签样本，从零开始训练:

```
python train.py --dataset cifar10 \
    --num-labeled 4000 \
    --arch wideresnet --batch-size 64 --lr 0.03 \
    --expand-labels --seed 5 \
    --out results/cifar10@4000.5 \
    --data-file path/to/cifar10
```

- 命令行2：从预训练模型继续训练，添加命令行参数 `--resume path/to/latest-ckpt`

```
python train.py --dataset cifar10 \
    --num-labeled 4000 \
    --arch wideresnet --batch-size 64 --lr 0.03 \
    --expand-labels --seed 5 \
    --out results/cifar10@4000.5 \
    --data-file path/to/cifar10 \
    --resume path/to/latest-ckpt 
```

### Test

- 方式1: 加载模型中保存的 dict:
```python
import paddle 
state=paddle.load("path/to/model_best.pdparams")
print(state.keys())  # 查看model中保存的dict
print(state['epoch'])  # 查看是哪个epoch保存的最佳模型
print(state['best_acc'])  # 查看模型的测试集精度
```

- 方式2: 使用命令行2，并添加命令行参数 `--eval_step 10`，以便更快地运行到 test 部分
```
python train.py --dataset cifar10 \
    --num-labeled 4000 \
    --arch wideresnet --batch-size 64 --lr 0.03 \
    --expand-labels --seed 5 \
    --out results/cifar10@4000.5 \
    --data-file path/to/cifar10 \
    --resume path/to/latest-ckpt \
    --eval_step 10
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
