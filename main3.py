import numpy as np
import torch as t
from progressbar import *
from torch import optim
from tqdm import tqdm
import os
import sys
import time
# from model import Model
# from model2 import Model
from model3 import Model
from utils.parser import load_args
import utils.tools as tl
import utils.handle_distance_mat as hdm
from utils.tools import cprint

class Run():
    def __init__(self, args):
        cprint("******************************** Choice device **********************************")
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        print("!!!", self.device, "is ready!!!")

        # 加载路径
        self.filepath = args.data_path + args.dataset + '/{}_{}'.format(args.domain_S, args.domain_T)
        # kaggle
        # self.kaggle_result_path = args.kaggle_result_path #kaggle

    def load_data(self):
        # 加载路径
        turnnum_path_S = self.filepath + '/general_turnnum_{}.json'.format(args.domain_S)
        turnnum_path_T = self.filepath + '/general_turnnum_{}.json'.format(args.domain_T)
        mat_path_S = self.filepath + '/new_{}_'.format(args.domain_S)
        mat_path_T = self.filepath + '/new_{}_'.format(args.domain_T)

        # 读取数据，开始训练第一步  common + distinct
        userNum_S, itemNum_S, user_item_index_S, n_interactions_S, real_item_S = tl.load_data(self.filepath + '/result_{}.dat'.format(args.domain_S))
        userNum_T, itemNum_T, user_item_index_T, n_interactions_T, real_item_T = tl.load_data(self.filepath + '/result_{}.dat'.format(args.domain_T))
        # set：
        # movie
        # music
        print(userNum_S, itemNum_S, n_interactions_S, real_item_S)
        print(userNum_T, itemNum_T, n_interactions_T, real_item_T)

        # 随机选择用于测试的数据，存储为一个字典
        ratings_test_S, ratings_train_S = tl.handle_tarin_test(user_item_index_S)
        ratings_test_T, ratings_train_T = tl.handle_tarin_test(user_item_index_T)
        print("!!! train and test are done !!!")

        # Generate the Laplacian matrix
        adj_mat_S, norm_adj_mat_S, mean_adj_mat_S, pre_adj_mat_S = tl.get_adj_mat(filepath=self.filepath, dataset=args.domain_S, n_users=userNum_S,
                                                   n_items=itemNum_S,
                                                   user_ratings=user_item_index_S,
                                                   user_ratings_test=ratings_test_S)
        adj_mat_T, norm_adj_mat_T, mean_adj_mat_T, pre_adj_mat_T = tl.get_adj_mat(filepath=self.filepath, dataset=args.domain_T, n_users=userNum_T,
                                                   n_items=itemNum_T,
                                                   user_ratings=user_item_index_T,
                                                   user_ratings_test=ratings_test_T)
        print("!!! Generate the Laplacian matrix is done !!!")

        i2i_mat_S, u2u_mat_S, ui_mat_S = hdm.get_distance_mat(mat_path_S, turnnum_path_S, userNum_S, itemNum_S, ratings_train_S)
        print("!!! create source domain distance mat is done !!!")
        i2i_mat_T, u2u_mat_T, ui_mat_T = hdm.get_distance_mat(mat_path_T, turnnum_path_T, userNum_T, itemNum_T, ratings_train_T)
        print("!!! create target domain distance mat is done !!!")

        i2i_mat_S = (i2i_mat_S != 0) * 1  # 将self.userDistanceMat中不为0的元素置为1
        u2u_mat_S = (u2u_mat_S != 0) * 1
        ui_mat_S = (ui_mat_S != 0) * 1
        i2i_mat_T = (i2i_mat_T != 0) * 1  # 将self.userDistanceMat中不为0的元素置为1
        u2u_mat_T = (u2u_mat_T != 0) * 1
        ui_mat_T = (ui_mat_T != 0) * 1

        # 封装成字典
        config = dict()  # dict()
        config['userNum_S'] = userNum_S
        config['userNum_T'] = userNum_T
        config['itemNum_S'] = itemNum_S
        config['itemNum_T'] = itemNum_T
        config['adj_mat_S'] = adj_mat_S
        config['adj_mat_T'] = adj_mat_T
        config['i2i_mat_S'] = i2i_mat_S
        config['i2i_mat_T'] = i2i_mat_T
        config['u2u_mat_S'] = u2u_mat_S
        config['u2u_mat_T'] = u2u_mat_T
        config['ui_mat_S'] = ui_mat_S
        config['ui_mat_T'] = ui_mat_T

        config['user_item_index_S'] = user_item_index_S
        config['user_item_index_T'] = user_item_index_T
        config['user_item_index_test_S'] = ratings_test_S
        config['user_item_index_test_T'] = ratings_test_T

        print("!!! load config is done !!!")

        return config

    # 准备模型
    def prepare_model(self, config):
        # 设置随机种子
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        t.cuda.manual_seed_all(args.seed)  # 如果使用多个GPU
        # t.backends.cudnn.deterministic = True
        # t.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        # 尝试设置 max_split_size_mb 以避免内存碎片问题
        # t.cuda.set_per_process_memory_fraction(0.35)

        self.model = Model(config, args, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    # 更新学习率
    def update_learning_rate(self):
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * args.decay, args.minlr)

    # 整体运行
    def overall_operation(self, config):
        self.prepare_model(config)

        best_ret_1 = t.tensor([0.0] * 4).to(self.device)
        best_ret_2 = t.tensor([0.0] * 4).to(self.device)

        # train
        print("!!! start train !!!")
        for epoch in range(args.epoch):

            t1 = time.time()
            epoch_loss = 0.
            epoch_loss2 = 0.

            # 显示进度条
            num_batches = len(config['user_item_index_S']) // args.batch_size  # 计算每个 epoch 中的批次数
            # 创建一个 tqdm 进度条，设置总迭代次数为 num_batches
            progress_bar = tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{args.epoch}', unit='batch', ncols=90)
            self.train = True

            for batch_idx in range(num_batches):

                # numpy数组，分别是：用户-项目交互字典中的用户索引、用户-项目交互字典中的项目索引、随机项目索引
                uij_1, uij_2 = tl.generate_train_batch_for_all_overlap(config, batch_size=args.batch_size)

                user1 = uij_1[:, 0]
                item_i1 = uij_1[:, 1]
                item_j1 = uij_1[:, 2]

                user2 = uij_2[:, 0]
                item_i2 = uij_2[:, 1]
                item_j2 = uij_2[:, 2]

                user_S = t.from_numpy(user1).long()
                item_i_S = t.from_numpy(item_i1).long()
                item_j_S = t.from_numpy(item_j1).long()
                itemindex_S = t.unique(t.cat((item_i_S, item_j_S)))
                userindex_S = t.unique(user_S)

                user_T = t.from_numpy(user2).long()
                item_i_T = t.from_numpy(item_i2).long()
                item_j_T = t.from_numpy(item_j2).long()
                itemindex_T = t.unique(t.cat((item_i_T, item_j_T)))
                userindex_T = t.unique(user_T)

                # 计算一个批次的损失
                train_result = self.model(user1, item_i1, item_j1, user2, item_i2, item_j2, userindex_S, itemindex_S,
                                       userindex_T, itemindex_T, self.train)

                loss = ((train_result['regLoss'] * args.reg) / num_batches) + train_result['ssl_loss_HetG'] * args.ssl_beta + \
                            train_result['metaregloss'] * args.metareg

                # total_loss = 0.65 * (train_result['bpr_loss'] + train_result['ssl_loss_MI']) + 0.35 * loss
                total_loss = 0.6 * train_result['DCCloss'] + 0.4 * loss

                # 它使用了梯度下降优化算法来更新模型的参数以减小损失函数
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()  # 使用优化器更新模型的参数
                epoch_loss += train_result['DCCloss'].item() / num_batches
                # epoch_loss += train_result['bpr_loss'].item() / num_batches
                # epoch_loss2 += train_result['bpr_loss'].item()
                # 更新进度条
                progress_bar.update(1)
                # progress_bar.set_postfix(loss = epoch_loss / (batch_idx + 1))  # 更新进度条的显示信息

            # 关闭进度条
            progress_bar.close()
            # 打印每个 epoch 的平均损失
            cprint(f'Epoch parsers: time:{time.time() - t1}s, Average Loss: {epoch_loss:.4f}')
            # cprint(f'Epoch parsers: time:{time.time() - t1}s, Average Loss: {epoch_loss2:.4f}')

            user_count_1 = 0
            user_count_2 = 0
            ret_1 = t.tensor([0.0] * 4).to(self.device)
            ret_2 = t.tensor([0.0] * 4).to(self.device)
            self.train = False

            # 生成用于测试的批次数据
            # num_batches = int(0.1 * len(config['user_item_index_S']))  # 计算每个 epoch 中的批次数
            num_batches = len(config['user_item_index_S'])
            test_pbar = tqdm(total=num_batches, desc=f'Testing {epoch + 1}/{args.epoch}', ncols=90)

            with test_pbar as progress_bar:
                for t_uij_1, t_uij_2 in tl.generate_test_batch_for_all_overlap(config):
                    t_user1 = t_uij_1[:, 0]
                    t_item_i1 = t_uij_1[:, 1]
                    t_item_j1 = t_uij_1[:, 2]

                    t_user2 = t_uij_2[:, 0]
                    t_item_i2 = t_uij_2[:, 1]
                    t_item_j2 = t_uij_2[:, 2]

                    self.model.get_embeddings(t_user1, t_item_i1, t_item_j1, t_user2, t_item_i2, t_item_j2)

                    u_g_embeddings_1, u_g_embeddings_2 = self.model.u_g_embeddings_1, self.model.u_g_embeddings_2
                    pos_i_g_embeddings_1, pos_i_g_embeddings_2 = self.model.pos_i_g_embeddings_1, self.model.pos_i_g_embeddings_2
                    neg_i_g_embeddings_1, neg_i_g_embeddings_2 = self.model.neg_i_g_embeddings_1, self.model.neg_i_g_embeddings_2

                    # 用于计算用户和物品之间的评分
                    pos_s_1 = t.squeeze(tl.rating(u_g_embeddings_1, pos_i_g_embeddings_1)).detach()
                    neg_s_1 = t.squeeze(tl.rating(u_g_embeddings_1, neg_i_g_embeddings_1)).detach()
                    pos_s_2 = t.squeeze(tl.rating(u_g_embeddings_2, pos_i_g_embeddings_2)).detach()
                    neg_s_2 = t.squeeze(tl.rating(u_g_embeddings_2, neg_i_g_embeddings_2)).detach()

                    # 将预测结果进行排序并计算排名
                    user_count_1 += 1
                    predictions_1 = pos_s_1[0]
                    predictions_1 = t.cat((predictions_1.view(1), neg_s_1))
                    predictions_1 = -1 * predictions_1
                    rank_1 = t.argsort(predictions_1)[0]
                    if rank_1 < 5:
                        ret_1[0] += 1
                        ret_1[2] += 1 / t.log2(rank_1 + 5)
                    if rank_1 < 10:
                        ret_1[1] += 1
                        ret_1[3] += 1 / t.log2(rank_1 + 10)

                    user_count_2 += 1
                    predictions_2 = pos_s_2[0]
                    predictions_2 = t.cat((predictions_2.view(1), neg_s_2))
                    predictions_2 = -1 * predictions_2
                    rank_2 = t.argsort(predictions_2)[0]
                    if rank_2 < 5:
                        ret_2[0] += 1
                        ret_2[2] += 1 / t.log2(rank_2 + 5)
                    if rank_2 < 10:
                        ret_2[1] += 1
                        ret_2[3] += 1 / t.log2(rank_2 + 10)

                    self.update_learning_rate()
                    # 更新测试进度条
                    progress_bar.update(1)

            # 关闭测试进度条
            test_pbar.close()

            best_ret_1 = tl.best_result(best_ret_1, ret_1)
            best_ret_2 = tl.best_result(best_ret_2, ret_2)

            print('%s: HR@5 %f HR@10 %f' % (
            args.domain_S, ret_1[0] / user_count_1, ret_1[1] / user_count_1))
            print('%s: NDCG@5 %f NDCG@10 %f' % (
            args.domain_S, ret_1[2] / user_count_1, ret_1[3] / user_count_1))

            print('Best HitRatio for %s: HR@5 %f HR@10 %f' % (
            args.domain_S, best_ret_1[0] / user_count_1, best_ret_1[1] / user_count_1))
            print('Best NDCG for %s: NDCG@5 %f NDCG@10 %f' % (
            args.domain_S, best_ret_1[2] / user_count_1, best_ret_1[3] / user_count_1))

            # if ret_1[0] == best_ret_1[0] or ret_1[1] == best_ret_1[1] or ret_1[2] == best_ret_1[2] or ret_1[3] == best_ret_1[3]:
            #     # tl.model_save(self.model.embedding_dict, args.weights_path, args, savename='best_model')
            #     print('save the weights in path: ', args.weights_path, 'save the domain: ', args.domain_S)


            print('%s: HR@5 %f HR@10 %f'
                  % (args.domain_T, ret_2[0] / user_count_2, ret_2[1] / user_count_2))
            print('%s: NDCG@5 %f NDCG@10 %f'
                  % (args.domain_T, ret_2[2] / user_count_2, ret_2[3] / user_count_2))
            print('Best HitRatio for %s: HR@5 %f HR@10 %f'
                  % (args.domain_T, best_ret_2[0] / user_count_2, best_ret_2[1] / user_count_2))
            print('Best NDCG for %s: NDCG@5 %f NDCG@10 %f'
                  % (args.domain_T, best_ret_2[2] / user_count_2, best_ret_2[3] / user_count_2))

            # if ret_2[0] == best_ret_2[0] or ret_2[1] == best_ret_2[1] or ret_2[2] == best_ret_2[2] or ret_2[3] == best_ret_2[3]:
            #     # model_save(model.embedding_dict, args.weights_path, args, savename='best_model')
            #     print('save the weights in path: ', args.weights_path, 'save the domain: ', args.domain_T)
            print('\n')


if __name__ == '__main__':
    # hyper parameters
    args = load_args()
    cprint("—————————————— Run with following settings ——————————————")
    # hyper parameters
    print(args)

    run = Run(args)
    config = run.load_data()
    run.overall_operation(config)