import d2l.torch as d2l
import torch
from model.alxnet import AlxNet




def eval_model(net, data_iter, device):


    return None


def train_model(net, train_iter, test_iter, lr, num_epoches, device):


    return None



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = AlxNet().to(device)
