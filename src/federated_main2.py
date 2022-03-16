
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
from utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu_id:
    # if args.gpu:
    #     torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('using {} for training'.format(device))

    # load dataset and user groups
    # 获取对应的数据集，并拆分成训练集（50000），测试集（10000）
    # 默认有100个联邦学习服务器，每一个服务器需要训练500条数据
    train_dataset, test_dataset, user_groups = get_dataset(args)
    print('train_dataset', train_dataset)
    print('test_dataset', test_dataset)
    print('group一共有', len(user_groups))
    print('每一个group有多少条数据', len(user_groups[0]))

    # BUILD MODEL
    # 根据参数建立对应的模型（卷积网络还是全连接网络）
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

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    # 外层循环表示一共联邦学习多少轮
    for epoch in tqdm(range(args.epochs)):
        # 使用local_weights和local_losses模拟每一轮所有训练节点上传自己得出的权重和损失
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch} |\n')

        global_model.train()
        # frac是一个小数，代表每一次抽取训练节点的数量，这里frac=0.1,表示抽取10%的节点来训练
        m = max(int(args.frac * args.num_users), 1)
        # 随机抽取
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # 对于每一个抽取到的训练节点
        for idx in idxs_users:
            # 训练节点需要进行本地模型更新，然后上传至中心节点聚合

            # 首先创建一个局部模型
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            # 这个训练节点对于自己的数据进行本地训练（一共--local_ep轮），得到
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)

            # 这里得到的w是一个字典结构，因为有多层网络，w将每一层网络之间的权重都表示出来
            # 例如 （'conv2.weight', ...） ('conv2.bias', ...) ('fc2.weight', ...)等

            # print(idx, '用户得到的w是: ', w)
            # print(idx, '用户得到的loss是: ', loss)  # loss就是一个数值

            # 训练节点将自己计算出的w和loss上传至中心服务器，即放到前面定义的两个字典里
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        # 中心服务器根据所有训练节点的权重来进行平均求解（逻辑就是求所有对应层参数的和的平均值）
        '''
        这里应该就是毕设可以突破的点
        可以把平均求解替换成距离比对权重求解
        '''
        global_weights = average_weights(local_weights)

        # update global weights
        # 将上面聚合的权重添加到模型中
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
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
            print("Node Training Acc: ", acc, "Loss: ", loss)
        # 把这一轮聚合后的训练准确度添加到变量train_accuracy中
        train_accuracy.append(sum(list_acc)/len(list_acc))


        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

            # filename = 'federal_weight.txt'
            # with open(filename, 'a') as file_object:
            #     file_object.write("\n\n\nepoch: " + str(epoch) +"\n\n\n")
            #     file_object.write(str(global_weights))
            #     file_object.write("\n\n\n")

    # Test inference after completion of training
    # filename = 'federal_weight.txt'
    # with open(filename, 'a') as file_object:
    #     file_object.write(str(global_weights))
    #     file_object.write("\n\n\n")
    # 到这里说明已经训练完毕，需要使用测试集对于上面训练的模型进行测试

    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    # 将本次训练的准确率和loss写入到文件中持久化
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

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
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
