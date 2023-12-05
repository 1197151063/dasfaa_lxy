import argparse


def load_args():
    parser = argparse.ArgumentParser(description='parse main.py')
    parser.add_argument('--data_path', nargs='?', default='./dataset/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='Amazon', help='Choose a dataset from {Amazon, XXX}.')
    parser.add_argument('--domain_S', nargs='?', default='movie', help='Choose a source domain.')
    parser.add_argument('--domain_T', nargs='?', default='music', help='Choose a target domain.')

    # 为测试稀疏性做准备
    parser.add_argument('--num', type=int, default=5, help='sparsity interaction number.')
    parser.add_argument('--score', type=float, default=4.0, help='sparsity retained score.')

    parser.add_argument('--seed', type=int, default=42, metavar='int', help='random seed')
    parser.add_argument('--A_split', type=bool, default=False, help='a_split')
    parser.add_argument('--keep_prob', type=float, default=0.6, help='keep_prob of dropout in lightgcn')
    parser.add_argument('--n_layers', type=int, default=3, help='Layer numbers.')
    parser.add_argument('--LayerNums', type=int, default=3, help='the numbers of uu-GCN layer')  # 要不要改成3
    parser.add_argument('--alpha', type=float, default=0.001, help='Number of epochs')
    parser.add_argument('--n_factors', type=int, default=2, help='Number of factors to disentangle the original embed-size representation.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Load iterative batches of data.')
    parser.add_argument('--size', type=int, default=8192, help='Load iterative batches of data.')
    parser.add_argument('--regs', nargs='?', default='[1e-3,1e-4,1e-4]', help='Regularizations.')  # useless
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    parser.add_argument('--hide_dim', type=int, default=64, metavar='N', help='embedding size')   # 要不要改成64
    parser.add_argument('--rank', type=int, default=3, help='the dimension of low rank matrix decomposition')

    # aggreation of the features of parameters
    parser.add_argument('--wu1', type=float, default=0.8, help='the coefficient of feature fusion ')  # 特征融合系数
    parser.add_argument('--wu2', type=float, default=0.2, help='the coefficient of feature fusion')
    parser.add_argument('--wi1', type=float, default=0.8, help='the coefficient of feature fusion ')
    parser.add_argument('--wi2', type=float, default=0.2, help='the coefficient of feature fusion')

    # 0.001 music 效果还行，在上升  movie 一直在下降
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learnidng rate')
    # parser.add_argument('--lr_d', type=float, default=0.055, help='Learning rate.')
    parser.add_argument('--minlr', type=float, default=0.0001)
    parser.add_argument('--decay', type=float, default=0.8, metavar='LR_decay', help='decay')
    parser.add_argument('--maxlr', type=float, default=0.6)
    parser.add_argument('--growth', type=float, default=1.1, metavar='LR_decay', help='decay')

    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
    # movie: 6000  garden: 4000
    parser.add_argument('--need_number', type=int, default=6000, help='Number of epochs')

    parser.add_argument('--ssl_ureg', type=float, default=0.04)
    parser.add_argument('--ssl_ireg', type=float, default=0.05)
    parser.add_argument('--ssl_temp_h', type=float, default=0.5, help='the temperature in softmax')
    parser.add_argument('--reg', type=float, default=0.043)  # default=0.043
    parser.add_argument('--ssl_beta', type=float, default=0.32, help='weight of loss with ssl')
    parser.add_argument('--metareg', type=float, default=0.15, help='weight of loss with reg')

    parser.add_argument('--weights_path', nargs='?', default='haven`t write',  help='Project path.')

    args = parser.parse_args()
    return args
