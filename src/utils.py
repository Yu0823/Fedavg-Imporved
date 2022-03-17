import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        # 对图像进行预处理 ToTensor: 将图像格式转化为tensor Normalize: 将图像正则化 Normalized_image=(image-mean)/std
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users 将数据拆分到不同的学习节点
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    # 这里w是所有训练节点产生的所有权重字典的数组
    # 这里的逻辑就是先遍历字典中所有的key，然后把所有对应key的所有权重相加，然后所有key对应的权重和再除以总数量
    # 相当于就是求平均值
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def global_weights_aggregate(w_center, w, idxs_nodes, abnormal_list, node_dis_last, distance_max, dis_inc_max):
    """
    :param w_center: the weight matrix of the center model
    :param w: the list of weight matrix of of other nodes' model
    :param idxs_nodes: the order list of nodes that need aggregate
    :param abnormal_list: the list of abnormal nodes
    :param node_dis_last: list of distance of a node model with center model in last round
    :param distance_max: the maximum threshold of the distance
    :param dis_inc_max the maximum distance increase percentage threshold
    Returns the weights according to the distance.
    """
    # 遍历每个key 代表了不同层的权重 / 偏差

    w_final = copy.deepcopy(w[0])
    distance = np.zeros(len(w))
    for key in w[0].keys():
        # 遍历每个节点上传的数据
        for i in range(0, len(w)):
            # 计算每个节点权重和中心节点权重的偏差
            distance[i] = distance[i] + get_distance_of_two_metrics(w_center[key], w[i][key])
        with open('log_for_debug.txt', 'a') as file_object:
            file_object.write("distance:")
            file_object.write(str(distance) + '\n')

    # 至此已经计算完此轮的距离

    w_node = np.zeros(len(w))   # w_node存储初步权重
    for i in range(0, len(w)):
        # 获取当前节点的真实序号
        node_num = idxs_nodes[i]
        with open('log_for_debug.txt', 'a') as file_object:
            file_object.write("node_num:" + str(node_num))
        # 判断是否大于距离最大阈值 如果大于则将此节点标记为异常节点
        if distance[i] > distance_max:
            # 加入异常节点名单
            abnormal_list.append(node_num)
            w_node[i] = 0
            with open('log_for_debug.txt', 'a') as file_object:
                file_object.write("!!!!!!!!!!!!\n")
        else:
            if node_num in node_dis_last:
                increase = distance[i] / node_dis_last[node_num]
                # 判断此节点与上一轮相比较的距离增长是否大于阈值 如果大于则将此节点标记为异常节点
                if increase > dis_inc_max:
                    abnormal_list.append(node_num)
                    w_node[i] = 0
                    with open('log_for_debug.txt', 'a') as file_object:
                        file_object.write("!!!!!!!!!!!!\n")
                else:
                    # 完全一致 基本不可能出现
                    if distance[i] == 0:
                        w_node[i] = 10
                    else:
                        w_node[i] = 1 / distance[i]
            else:
                # 完全一致 基本不可能出现
                if distance[i] == 0:
                    w_node[i] = 10
                else:
                    w_node[i] = 1 / distance[i]

        # 把该节点本次训练的距离加入到字典中
        node_dis_last[node_num] = distance[i]

    w_node_sum = w_node.sum()

    # 加权平均
    for key in w[0].keys():
        torch.zeros_like(w_final[key])
        for i in range(0, len(w)):
            w_final[key] = w_final[key] + w_node[i] / w_node_sum * w[i][key]

    return w_final, abnormal_list


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')

    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def get_distance_of_two_metrics(w1, w2):
    diff_matrix = w1 - w2
    shape = w1.shape
    length = len(shape)
    distance = 0
    if length == 1:
        for i in range(0, shape[0]):
            distance += diff_matrix[i] ** 2
    if length == 2:
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                distance += diff_matrix[i, j] ** 2
    if length == 3:
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    distance += diff_matrix[i, j, k] ** 2
    if length == 4:
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    for m in range(0, shape[3]):
                        distance += diff_matrix[i, j, k, m] ** 2
    return distance
