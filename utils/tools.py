import random as rd
from collections import defaultdict
from datetime import datetime

import scipy.sparse as sp
import torch as t
import numpy as np
import torch.nn.functional as F
import sys

from utils.parser import load_args
args = load_args()

# CYJ 彩虹框
def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

# 创建批次
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

# 获取用户总数和项目总数，userNum和itemNum
def calculate_total(data):
    # 初始化 userNum 为 0
    userNum = 0
    # 遍历数据，查找最大的 userID 值
    for record in data:
        userID = record["userID"]
        if userID > userNum:
            userNum = userID
    # 将最大值加 1 赋值给 userNum
    userNum += 1
    return userNum

# 随机选择用于测试的数据
def handle_tarin_test(data):
    # 用于存储每个用户的测试数据
    ratings_test = {}
    ratings_train = {}
    for user in data:
        ratings_test[user] = rd.sample(data[user], 1)[0]
        list = []
        for i in data[user]:
            if i != ratings_test[user]:
                list.append(i)
            else:
                continue
        ratings_train[user] = list
    return ratings_test, ratings_train


# 计算两个时间戳之间的距离
def compute_distance(timestamp_i, timestamp_j):
    # 将时间戳转换为datetime对象
    time_i = datetime.utcfromtimestamp(timestamp_i)
    time_j = datetime.utcfromtimestamp(timestamp_j)
    # 计算时间差，可以使用不同的时间差度量，这里使用绝对时间差
    time_difference = abs(time_i - time_j)
    # 将时间差表示为距离，例如，可以使用秒、分钟、小时等不同的时间单位
    # 在这里，我们将时间差表示为秒
    distance = time_difference.total_seconds()
    # 标准化距离到0到1之间
    max_seconds = 24 * 60 * 60  # 一天的秒数
    normalized_distance = min(distance / max_seconds, 1.0)
    return normalized_distance


def generate_train_batch_for_all_overlap(config, batch_size):
    user_ratings_1 = config['user_item_index_S']
    user_ratings_test_1 = config['user_item_index_test_S']
    n_1 = config['itemNum_S']
    user_ratings_2 = config['user_item_index_T']
    user_ratings_test_2 = config['user_item_index_test_T']
    n_2 = config['itemNum_T']

    t_1 = []
    t_2 = []
    for b in range(batch_size):
        u = rd.sample(user_ratings_1.keys(), 1)[0]
        i_1 = rd.sample(user_ratings_1[u], 1)[0]
        i_2 = rd.sample(user_ratings_2[u], 1)[0]
        while i_1 == user_ratings_test_1[u]:
            i_1 = rd.sample(user_ratings_1[u], 1)[0]
        while i_2 == user_ratings_test_2[u]:
            i_2 = rd.sample(user_ratings_2[u], 1)[0]
        j_1 = rd.randint(0, n_1 - 1)
        j_2 = rd.randint(0, n_2 - 1)
        while j_1 in user_ratings_1[u]:
            j_1 = rd.randint(0, n_1 - 1)
        while j_2 in user_ratings_2[u]:
            j_2 = rd.randint(0, n_2 - 1)
        t_1.append([u, i_1, j_1])
        t_2.append([u, i_2, j_2])
    train_batch_1 = np.asarray(t_1)  # 将列表转化为nuppy数组
    train_batch_2 = np.asarray(t_2)
    return train_batch_1, train_batch_2


# Contrastive Learning
def ssl_loss(data1, data2, index):
    # 是一个包含索引的张量，它用于指示应该在输入数据中选择哪些样本进行计算
    index = t.from_numpy(index)
    index=t.unique(index)  # 首先确保索引 index 中的值不重复，这可以避免重复计算相同的样本对
    embeddings1 = data1[index]
    embeddings2 = data2[index]
    norm_embeddings1 = F.normalize(embeddings1, p = 2, dim = 1)
    norm_embeddings2 = F.normalize(embeddings2, p = 2, dim = 1)
    pos_score  = t.sum(t.mul(norm_embeddings1, norm_embeddings2), dim = 1)  # 正样本对的相似度分数，这里采用了向量点积来度量它们之间的相似性。
    all_score  = t.mm(norm_embeddings1, norm_embeddings2.T)  # 所有样本对的相似度分数矩阵，其中的每个元素表示一个样本对的相似性。
    pos_score  = t.exp(pos_score / args.ssl_temp)  # 应用了 softmax 操作，这有助于将分数转化为概率分布
    all_score  = t.sum(t.exp(all_score / args.ssl_temp), dim = 1)
    # 目标是使正样本对的相似度分数更大，而将其他样本对的相似度分数尽可能地降低
    ssl_loss  = (-t.sum(t.log(pos_score / ((all_score))))/(len(index)))
    return ssl_loss


def predictModel(user, pos_i, neg_j, isTest=False):
    if isTest:
        pred_pos = t.sum(user * pos_i, dim=1)
        return pred_pos
    else:
        pred_pos = t.sum(user * pos_i, dim=1)
        pred_neg = t.sum(user * neg_j, dim=1)
        return pred_pos, pred_neg


# 生成用于测试的批次数据
def generate_test_batch_for_all_overlap(config):
    user_ratings_1 = config['user_item_index_S']
    user_ratings_test_1 = config['user_item_index_test_S']
    n_1 = config['itemNum_S']
    user_ratings_2 = config['user_item_index_T']
    user_ratings_test_2 = config['user_item_index_test_T']
    n_2 = config['itemNum_T']

    # test_id = int(0.1 * len(user_ratings_1))

    for u in user_ratings_1.keys():
    # if u < test_id:
        t_1 = []
        t_2 = []
        i_1 = user_ratings_test_1[u]
        i_2 = user_ratings_test_2[u]
        rated_1 = user_ratings_1[u]
        rated_2 = user_ratings_2[u]
        for j in range(999):
            k = np.random.randint(0, n_1-1)
            while k in rated_1:
                k = np.random.randint(0, n_1-1)
            t_1.append([u, i_1, k])
        for j in range(999):
            k = np.random.randint(0, n_2-1)
            while k in rated_2:
                k = np.random.randint(0, n_2-1)
            t_2.append([u, i_2, k])
        yield np.asarray(t_1), np.asarray(t_2)
    # else:
    #     break

# 用于计算用户和物品之间的评分
def rating(u_g_embeddings, i_g_embeddings):
    return t.sigmoid(t.sum(u_g_embeddings*i_g_embeddings, axis=1, keepdim=True))

# 将两个列表中对应位置的元素进行比较，取较大的值组成一个新的列表，并返回该列表。
def best_result(best, current):
    # print("find the best number:")
    # num_ret = len(best)
    # ret_best = [0.0]*num_ret
    # for numIdx in range(num_ret):
    #     ret_best[numIdx] = max(float(current[numIdx]), float(best[numIdx]))
    # return ret_best
    return t.maximum(best, current)

def model_save(weights, path, args, savename='best_model'):
    save_pretrain_path = '%spretrain/%s/%s' % (path, args.dataset+'_'+args.domain_1+'_'+args.domain_2, savename)
    np.savez(save_pretrain_path, user_embed_1=weights['user_embedding_1'].detach().cpu().numpy(),
                                 item_embed_1=weights['item_embedding_1'].detach().cpu().numpy(),
                                 user_embed_2=weights['user_embedding_2'].detach().cpu().numpy(),
                                 item_embed_2=weights['item_embedding_2'].detach().cpu().numpy())


def load_data(filepath):
    # 第一个是用户ID，其余的是项目ID，一行的长度就是该用户的交互数量，所有行长度的值即为域中用户项目交互数量总数
    n_users = 0
    n_items = 0
    n_interactions = 0  # 交互数量
    # 创建一个空的集合来存储itemID
    item_set = set()
    with open(filepath) as f:
        for l in f.readlines():
            if len(l) > 0:
                n_users += 1
                l = l.strip('\n').rstrip(' ')
                items = [int(i) for i in l.split(' ')]
                n_items = max(n_items, max(items))
                n_interactions += (len(items)-1)
                parts = l.strip().split()
                # 遍历itemID列表
                for item_id in parts[1:]:
                    # 将itemID添加到集合中
                    item_set.add(item_id)
    # 计算item的数量，即集合的大小
    item_num = len(item_set)
    n_items += 1
    user_ratings = defaultdict(dict)
    with open(filepath) as f:
        for l in f.readlines():
            if len(l) == 0:
                break
            l = l.strip('\n').rstrip(' ')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]
            # 键是用户ID，值是用户与物品之间的交互数据列表。
            user_ratings[uid] = train_items
    return n_users, n_items, user_ratings, n_interactions, item_num


def get_adj_mat(filepath, dataset, n_users, n_items, user_ratings, user_ratings_test):
    try:
        # 邻接矩阵、规范化邻接矩阵和均值邻接矩阵
        adj_mat = sp.load_npz(filepath + '/{}_adj_mat.npz'.format(dataset))
        norm_adj_mat = sp.load_npz(filepath + '/{}_norm_adj_mat.npz'.format(dataset))
        mean_adj_mat = sp.load_npz(filepath + '/{}_mean_adj_mat.npz'.format(dataset))
        print('already load adj matrix', adj_mat.shape)

    except Exception:
        adj_mat, norm_adj_mat, mean_adj_mat = create_adj_mat(n_users, n_items, user_ratings, user_ratings_test)
        sp.save_npz(filepath + '/{}_adj_mat.npz'.format(dataset), adj_mat)
        sp.save_npz(filepath + '/{}_norm_adj_mat.npz'.format(dataset), norm_adj_mat)
        sp.save_npz(filepath + '/{}_mean_adj_mat.npz'.format(dataset), mean_adj_mat)

    try:
        # 预处理邻接矩阵
        pre_adj_mat = sp.load_npz(filepath + '/{}_pre_adj_mat.npz'.format(dataset))
    except Exception:
        adj_mat=adj_mat
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv)
        print('generate pre adjacency matrix.')
        pre_adj_mat = norm_adj.tocsr()
        sp.save_npz(filepath + '/{}_pre_adj_mat.npz'.format(dataset), norm_adj)

    return adj_mat, norm_adj_mat, mean_adj_mat, pre_adj_mat


# 创建一个稀疏邻接矩阵，用于表示用户-物品之间的交互关系
def create_adj_mat(n_users, n_items, user_ratings, user_ratings_test):
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()  # 首先创建一个字典形式的稀疏矩阵，然后将其转换为 LIL 格式（列表列表格式）。

    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)  # 表示用户与物品之间的交互关系（排除测试项目，有交互即为1）
    for uid in user_ratings.keys():
        for item in user_ratings[uid]:
            if not item == user_ratings_test[uid]:
                R[uid, item] = 1
    R = R.tolil()

    # 将创建的 R 矩阵分别填充到邻接矩阵的左上角和右下角
    # 上半部分表示用户到物品的连接，下半部分表示物品到用户的连接。
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape)  # 打印已经创建的邻接矩阵的形状，即行数和列数。

    # 对邻接矩阵进行单一归一化处理
    def normalized_adj_single(adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        # tocoo()这个转换可能在需要逐元素处理稀疏矩阵时有用
        return norm_adj.tocoo()

# 将邻接矩阵和单位阵相加，通常用于构建图的拉普拉斯矩阵或其他图上的变换
    norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    mean_adj_mat = normalized_adj_single(adj_mat)

    print('already normalize adjacency matrix')

    # 在 COO 格式中，稀疏矩阵的非零元素被存储为三个分别表示行索引、列索引和元素值的数组。
    # 在 CSR 格式中，数据被分别存储为行指针、列索引和元素值的数组，以压缩表示。
    # tocsr()这个转换通常用于优化稀疏矩阵的内存占用和计算性能，特别是在矩阵向量乘法等操作中。
    # tocoo()这个转换可能在需要逐元素处理稀疏矩阵时有用
    return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()