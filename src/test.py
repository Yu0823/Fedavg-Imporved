import torch
import numpy as np
import utils
import pickle

if __name__ == '__main__':
    data = pickle.load(open('../save/objects/mnist_cnn_10_C[1.0]_iid[0]_E[5]_B[10].pkl', 'rb'))  # 记得加上'rb'
    print(data)
    # a = [1, 2, 3, 4, 5]
    # b = [4, 1]
    # c = [i for i in a if i not in b]
    # print(c)
    # a = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    # b = torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]])
    # c = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 1.0, 3.0]])
    # w_c = {"k1": a, "k2": b, "k3": c}
    # w = [{"k1": a, "k2": b, "k3": c}]
    # print("w_c:", w_c, "\nw_c", w)
    # w_f = utils.calculate_weights_by_distance(w_c, w)
    # print(w_f)
    # print("a:", a, "b:", b)
    # d = utils.get_distance_of_two_metrics(a, b)
    # print("d:", d)
    # data = a[1, 1]
    # print(data)
    # distance = np.zeros(10)
    # print(distance)
    # sum1 = np.sum(a.numpy)
    # print(sum1)

