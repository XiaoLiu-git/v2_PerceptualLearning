import os
import numpy as np

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize(x):
    mean = np.expand_dims(np.mean(x, axis=1), 1)
    sigma = np.expand_dims(np.std(x, axis=1), 1)
    sigma[sigma == 0] = 1
    nor_x = (x - mean) / sigma
    return nor_x

def np_acc(outputs, labels):
    output = outputs.detach().numpy()
    total = (np.sign(output * labels) + 1) / 2
    acc = np.sum(total) / len(output)
    return acc

