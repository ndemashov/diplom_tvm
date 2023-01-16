import tvm
from tvm.relay import Call
from tvm import relax, tir
from tvm.relax.testing import nn
from tvm.script import relax as R
import numpy as np

import pandas as pd

dtype = "float32"
l1 = np.loadtxt("weights/dense.csv", delimiter=",").astype(dtype)
#l1 = l1.transpose(1, 0)
print("linear1", l1.shape)
b1 = np.loadtxt("weights/bias.csv", delimiter=",").astype(dtype)
print("bias1", b1.shape)
l2 = np.loadtxt("weights/dense_1.csv", delimiter=",").astype(dtype)
#l2 = l2.transpose(1, 0)
print("linear2", l2.shape)
b2 = np.loadtxt("weights/bias_1.csv", delimiter=",").astype(dtype)
print("bias1", b2.shape)

from skimage import io

def read_image(path):
  image = io.imread(path)
  return image

def one_hot_encoding(Y):
    classes = np.reshape(np.unique(Y),(-1,1))
    Y_enc = np.equal(classes,Y)*1
    return Y_enc

def eval():
    train_data = pd.read_csv('mnist_test.csv')
    np_data = np.array(train_data)
    X_train = np_data[:,1:].T # All except label col [0]
    X_train = X_train/255 # Image normalize
    Y_train = np.reshape(np_data[:,0],(1,-1))
    Y_train = one_hot_encoding(Y_train)
    for i in range(100, 110):
        image = X_train[:, i]
        label = Y_train[:, i]
        image = image.reshape(1, 28*28)
        image = image.astype(dtype) / 255
        data_nd = tvm.nd.array(image)
        nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

        params = [data_nd, nd_params["w0"], nd_params["b0"], nd_params["w1"], nd_params["b1"]]
        res = vm["main"](*params)
        print(res)
        pred_kind = np.argmax(res.numpy(), axis=1)
        print("Predicted number: {}".format(class_names[pred_kind[0]]), np.argmax(label))

if __name__ == "__main__":
    builder = relax.BlockBuilder()

    # a symbolic variable to represent minibatch size
    n = 1
    input_size = 784
    hidden_sizes = [512]
    output_size = 10

    # build a three linear-layer neural network for a classification task
    with builder.function("main"):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], output_size),
        )
        data = nn.Placeholder((n, input_size), name="data")
        output = model(data)
        params = [data] + model.parameters()
        builder.emit_func_output(output, params=params)

    # get and print the IRmodule being built
    mod = builder.get()
    mod.show()

    # build the IRModule and create relax vm
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    mlp_params = {}
    mlp_params["w0"] = l1
    mlp_params["b0"] = b1
    mlp_params["w1"] = l2
    mlp_params["b1"] = b2

    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    test_image_choice = '9'
    image_path = "img/mnist_img_{}.jpg".format(test_image_choice)
    image = read_image(image_path)
    image = image.reshape(1, 28*28)
    print(image.shape)
    image = image.astype(dtype) / 255
    data_nd = tvm.nd.array(image)
    nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}
    params = [data_nd, nd_params["w0"], nd_params["b0"], nd_params["w1"], nd_params["b1"]]
    res = vm["main"](*params)
    print(res)
    pred_kind = np.argmax(res.numpy(), axis=1)
    print("Predicted number: {}".format(class_names[pred_kind[0]]))

    eval()
