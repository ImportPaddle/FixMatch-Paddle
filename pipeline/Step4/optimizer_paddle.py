import paddle.optimizer as optim
from pipeline.utils import *

def gen_optimizer(model):
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0005},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    model_params_1=[p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)]
    optimizer_1 = optim.Momentum(learning_rate=0.03,momentum=0.9,weight_decay=0.0005,
                               parameters=model_params_1, use_nesterov=True)
    model_params_2=[p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)]
    optimizer_2 = optim.Momentum(learning_rate=0.03, momentum=0.9, weight_decay=0.0,
                                 parameters=model_params_2, use_nesterov=True)
    return optimizer_1,optimizer_2

if __name__ == '__main__':
    model_name='wideresnet'
    model_paddle,_=gen_model(model_name)
    optimizer=gen_optimizer(model_paddle)
    for i in range(10000):
        print(i,':optimizer:',optimizer[1]._learning_rate)