import torch.optim as optim
from pipeline.utils import *

def gen_optimizer(model):
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0005},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=0.03,
                          momentum=0.9, nesterov=True)
    return optimizer
if __name__ == '__main__':
    model_name='wideresnet'
    _,model_torch=gen_model(model_name)
    optimizer=gen_optimizer(model_torch)
    print('optimizer:',optimizer)