import paddle
from paddle.vision import transforms
from wide_resnet import WideResNet
import time
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=Warning)

paddle.set_device('gpu')
place = paddle.CUDAPlace(0)
model=WideResNet(28,10,20,0.3)


mean,std = ([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
mean = list(map(lambda x:x*255,mean))
std = list(map(lambda x:x*255,std))
val_loader = paddle.io.DataLoader(
    paddle.vision.datasets.Cifar10(mode='test', transform=transforms.Compose([
        transforms.Transpose(order=(2,0,1)),
        transforms.Normalize(mean=mean,std=std),
    ])), places=place,
    batch_size=256, shuffle=False,
    num_workers=4, use_shared_memory=True)

checkpoint=paddle.load('/home/aistudio/checkpoint.pdparams')
model.set_state_dict(checkpoint)
loss_fn = paddle.nn.CrossEntropyLoss()
acc_fn = paddle.metric.accuracy
accuracies=[]
losses=[]
model.eval()
total = 0
acc = 0

for (x,y) in val_loader:
    with paddle.no_grad():
        logits=model(x)
        y=paddle.reshape(y,(-1,1))
        loss=loss_fn(logits,y)
        logits = logits.numpy()
        y = y.numpy()
        total += len(logits)
        for index,item in enumerate(logits):
            label = np.argmax(item)
            if label == y[index][0]:
                acc += 1

print("acc:{} total:{} ratio:{}".format(acc,total,acc*1.0/total))