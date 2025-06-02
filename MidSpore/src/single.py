import os
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops
from mindspore.communication import init
from time import time

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_context(max_device_memory="8GB")
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()
ms.set_seed(1)

from mindspore import nn, ops
from mindspore.common.initializer import initializer
import mindspore as ms

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.weight1 = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.weight2 = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.weight3 = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        
        self.matmul1 = ops.MatMul()
        self.matmul2 = ops.MatMul()
        self.matmul3 = ops.MatMul()
        
        self.relu = ops.ReLU()
        self.softmax = ops.Softmax()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.weight1)
        x = self.relu(x)
        x = self.matmul2(x, self.weight2)
        x = self.relu(x)
        x = self.matmul3(x, self.weight3)
        logits = self.softmax(x)
        return logits

net = Network()


def create_dataset(batch_size):
    """create dataset"""
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

data_set = create_dataset(32)
optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    """forward propagation"""
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

@ms.jit
def train_step(inputs, targets):
    """train_step"""
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

start = time()
for epoch in range(1, 11):
    i = 0
    for image, label in data_set:
        loss_output = train_step(image, label)
        i += 1
        if i % 1875 == 0 and i != 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))

end = time()
print(end - start)