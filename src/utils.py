import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, cifar_noniid_unequal


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
                user_groups = cifar_noniid_unequal(train_dataset, args.num_users)
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
        # for i in range(1, len(w)):
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def global_weights_aggregate(w_center, w, node_acc_dic, idxs_nodes, abnormal_list, node_acc_last, args, record_filename):
    """
    :param w_center: the weight matrix of the center model
    :param w: the list of weight matrix of of other nodes' model
    :param node_acc_dic: the dict contains acc of all nodes in train
    :param idxs_nodes: the order list of nodes that need aggregate
    :param abnormal_list: the list of abnormal nodes
    :param node_acc_last: list of accuracy of a node model with center model in last round
    :param args: the args list
    :param record_filename the file to record training
    Returns the weights the final model after execute the algorithm.
    """
    # 遍历每个key 代表了不同层的权重 / 偏差
    w_final = copy.deepcopy(w[0])
    distance = np.zeros(len(w))
    for key in w[0].keys():
        # 遍历每个节点上传的数据
        for i in range(0, len(w)):
            distance[i] = distance[i] + get_distance_of_two_metrics(w_center[key], w[i][key])
    # 至此已经计算完此轮的距离

    # 进行异常节点处理
    for i in range(0, len(w)):
        node_num = idxs_nodes[i]
        if node_acc_dic[node_num] < args.acc_min:
            # 加入异常节点名单
            abnormal_list.append(node_num)
            with open(record_filename, 'a') as file_object:
                file_object.write("add node" + str(node_num) + " to abnormal list for node_acc value\n")
        elif node_num in node_acc_last:
            acc_drop = node_acc_last[node_num] - node_acc_dic[i]
            if acc_drop > args.acc_drop_max:
                abnormal_list.append(node_num)
                with open(record_filename, 'a') as file_object:
                    file_object.write("add node" + str(node_num) + " to abnormal list for node_acc drop\n")

    node_dis_rev = [0 for i in range(0, len(idxs_nodes))]  # 用于存储距离的倒数
    node_dis_rev_sum = 0
    w_node_dis = [0 for i in range(0, len(idxs_nodes))]  # w_node_dis用于存储用距离计算出的权重

    # 遍历得到距离的倒数
    for i in range(0, len(w)):
        if idxs_nodes[i] not in abnormal_list:
            if distance[i] == 0:
                node_dis_rev[i] = 100
            else:
                node_dis_rev[i] = 1 / distance[i]
            node_dis_rev_sum = node_dis_rev_sum + node_dis_rev[i]

    # 计算出基于距离的权重
    for i in range(0, len(w)):
        if idxs_nodes[i] not in abnormal_list:
            w_node_dis[i] = node_dis_rev[i] / node_dis_rev_sum

    w_node_agg = []

    # 生成节点最终聚合的权重
    node_acc_sum = 0
    for i in range(0, len(w)):
        if idxs_nodes[i] not in abnormal_list:
            node_acc_sum = node_acc_sum + node_acc_dic[idxs_nodes[i]]

    for i in range(0, len(w)):
        if idxs_nodes[i] not in abnormal_list:
            w_node_agg.append((node_acc_dic[idxs_nodes[i]] / node_acc_sum) * (1 - args.k) + w_node_dis[i] * args.k)
        else:
            w_node_agg.append(0)

    # 加权平均聚合
    for key in w[0].keys():
        w_final[key] = torch.zeros_like(w_final[key])
        for i in range(0, len(w)):
            w_final[key] = w_final[key] + w[i][key] * w_node_agg[i]

    test_sum = 0
    for i in range(0, len(w)):
        test_sum = test_sum + w_node_agg[i]

    with open(record_filename, 'a') as file_object:
        file_object.write("w_node_dis_sum:")
        file_object.write(str(node_dis_rev_sum))
        file_object.write(" node_acc_sum:")
        file_object.write(str(node_acc_sum))
        file_object.write(" final weight dic:")
        file_object.write(str(w_node_agg))
        file_object.write(" sum of weight:")
        file_object.write(str(test_sum))

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
