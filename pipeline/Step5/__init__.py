import torch
import paddle

a = torch.randn(4, 4)
print('a',a)
print(torch.max(a, -1))
b=paddle.randn([4,4])
print('b:',b)
print(paddle.max(b, -1))
a=torch.ones(2)
b=torch.zeros(2)
print(a)
print(a.ge(b))