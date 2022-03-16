import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, global_weights_aggregate
from update import center_update

if __name__ == '__main__':

    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    # path for log
    logger = SummaryWriter('../logs')

    # parse the running args
    args = args_parser()
    # print details of experiment
    exp_details(args)

    # choose the training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using {} for training'.format(device))

    # load dataset and user groups
    # 获取并处理对应的数据集，并拆分成训练集（50000）和测试集（10000）
    # 默认有100个联邦学习服务器，每一个服务器需要训练500条数据
    train_dataset, test_dataset, user_groups = get_dataset(args)
    print('train_dataset', train_dataset)
    print('test_dataset', test_dataset)
    print('group一共有', len(user_groups))
    print('每一个group有多少条数据', len(user_groups[0]))

    # BUILD MODEL
    # 根据参数建立对应的模型（卷积网络还是全连接网络） 包含详细的网络结构
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1

        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    abnormal_list = []
    node_dis_last = {3: 0.0001}

    # 外层循环表示一共联邦学习多少轮
    for epoch in tqdm(range(args.epochs)):
        # 使用local_weights和local_losses模拟每一轮所有训练节点上传自己得出的权重和损失
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch} |\n')

        # 训练中心节点的模型
        global_model_before = copy.deepcopy(global_model)
        global_model.train()
        center_update(global_model, args)
        center_weights = global_model.state_dict()

        # frac是一个小数，代表每一次抽取训练节点的比例 如frac=0.1表示抽取10%的节点来训练
        num_user_need = max(int(args.frac * args.num_users), 1)
        # 随机抽取训练节点 不再抽取在abnormal_list中已经被标记为异常的节点
        valid_users = [i for i in range(args.num_users) if i not in abnormal_list]
        # 如果异常节点数量多导致可选节点过少 则减小需要训练节点的数量
        if len(valid_users) == 0:
            exit('No valid users anymore!')
        if num_user_need > len(valid_users):
            num_user_need = len(valid_users)
        idxs_users = np.random.choice(valid_users, num_user_need, replace=False)
        filename = 'log_for_debug.txt'
        with open(filename, 'a') as file_object:
            file_object.write("epoch:")
            file_object.write(str(epoch) + '\n')
            file_object.write("node_dis_last:")
            file_object.write(str(node_dis_last) + '\n')
            file_object.write("idxs_users:")
            file_object.write(str(idxs_users) + '\n')

        # 对于每一个抽取到的训练节点 需要进行本地模型更新，然后上传至中心节点聚合
        for idx in idxs_users:
            # 首先创建一个局部更新
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)

            # 这个训练节点对于自己的数据进行本地训练（一共--local_ep轮），得到
            print('Global Round: {} Training on node {}:'.format(epoch, idx))
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model_before), global_round=epoch)

            # 这里得到的w是一个字典结构，因为有多层网络，w将每一层网络之间的权重都表示出来
            # 例如 （'conv2.weight', ...） ('conv2.bias', ...) ('fc2.weight', ...)等

            # print(idx, '用户得到的w是: ', w)
            # print(idx, '用户得到的loss是: ', loss)  # loss就是一个数值

            # 训练节点将自己计算出的w和loss上传至中心服务器，即放到前面定义的两个字典里
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights 运用设计的算法进行聚合 得到此轮的中心模型权重并更新异常节点列表
        global_weights, abnormal_list = global_weights_aggregate(center_weights, local_weights,
                                                                 idxs_users, abnormal_list, node_dis_last, args.dis_max,
                                                                 args.dis_inc_max)

        # update global weights 将上面聚合的权重添加到模型中
        global_model.load_state_dict(global_weights)

        # 再求取所有训练节点loss的平均值作为总体的loss
        loss_avg = sum(local_losses) / len(local_losses)

        # 把这一次聚合的loss添加到一个变量train_loss中
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            # 使用LocalUpdate对象中的inference方法来进行训练内的验证（类似于交叉验证，分配比例是8：1：1）
            # TODO 这里发现每个user调用此方法进行验证后acc和loss都一样 因为验证的是同一个model？
            acc, loss = local_model.inference(model=global_model_before)
            # print("Node Training Acc: ", acc, "Loss: ", loss)
            list_acc.append(acc)
            list_loss.append(loss)
        # 把这一轮聚合后的训练准确度添加到变量train_accuracy中
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Test inference after completion of training 训练完毕，需要使用测试集对于上面训练的模型进行测试

    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    # 将本次训练的准确率和loss写入到文件中持久化
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_ipv_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_ipv_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
