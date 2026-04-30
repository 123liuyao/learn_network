import torch
import torch.nn as nn
import d2l.torch as d2l
from model.lenet import LeNet
import matplotlib.pyplot as plt

def evaluate(net, data_iter, device):
    net.to(device)
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else: X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0]/metric[1]


def train_fuxi(net, train_iter, test_iter, lr, num_epochs, device):

    #首先进行参数初始化
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net.to(device)
    # 然后定义优化器 损失函数 数据存储
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train_loss', 'train_acc', 'test_acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    #开始训练
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l*X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]
            if (i + 1) % (num_batches // 5) ==0 or i == num_batches - 1:
                animator.add(epoch+(i+1)/num_batches, [train_l, train_acc, None])
        test_acc = evaluate(net, test_iter, device)
        animator.add(epoch+1, [None, None, test_acc])
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

if __name__ == '__main__':
    net = LeNet()
    if torch.cuda.is_available():
        device = 'cuda'
    else: device = 'cpu'
    train_iter, test_iter  = d2l.load_data_fashion_mnist(256)
    train_fuxi(net, train_iter, test_iter, 0.9, 10, device)
    plt.show()
