# Federated-Learning-System

Based on https://github.com/AshwinRJ/Federated-Learning-PyTorch

MNIST, Fashion MNIST and CIFAR10 are supported (both IID and non-IID). In case of non-IID, the data amongst the nodes can be split equally or unequally.

Since the initial purpose of the project is to applying the filtering and optimization strategy in federated study process and do some experiments, only simple models such as MLP and CNN are supported.

## Requirments
* Python 3.8
* Pytorch 1.7.1
* Numpy 1.19.2  

GPU is optinal
## Data
* Mnist, Fashion Mnist, and Cifar are supported
* They will be automatically downloaded from torchvision datasets, or you could download it manually
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments  
Three modes:
* baseline: traditional deep learning
* federated: use FedAvg to aggregate
* federated_improved: use the proposed strategy

The baseline experiment trains the model in the conventional way.
* To run the baseline experiment with MNIST on MLP using CPU:
```
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
```
-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with MNIST on CNN (IID):
```
python src/federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=1 --epochs=10
```
* To run the same experiment under non-IID condition:
```
python src/federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --epochs=10
```
-----
Federated_improve experiment use the proposed strategy.

* To run the federated experiment with MNIST on CNN (IID):
```
python src/federated_improved.py --model=cnn --dataset=mnist --gpu=0 --iid=1 --epochs=10
```
* To run the same experiment under non-IID condition:
```
python src/federated_improved.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --epochs=10
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
You could add following parameters in command-line to choose the option.
```
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--k', type=float, default=0.1,
                        help="the percentage of distance in weight counting")
    parser.add_argument('--acc_min', type=float, default=0.15,
                        help="the minimum threshold of the accuracy")
    parser.add_argument('--acc_drop_max', type=float, default=50,
                        help="the maximum accuracy drop percentage threshold")
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
```