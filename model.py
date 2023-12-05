import timeit

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

class Model(nn.Module):

    # parameter:一个用来存储所用实体的字典（userNum、itemNum等等）
    # args:用来调用parser的参数，方便在命令行修改

    def __init__(self, config, args, device):
        super(Model, self).__init__()
        ...
        # 初始化值
        self.device = device

        self.folds = 10
        self.ssl_temp = 0.1
        self.dropout = False

        self.A_split = args.A_split
        # keep_prob 参数表示保留神经元的概率，其值在 0 到 1 之间，通常是一个较小的值。
        self.keep_prob = args.keep_prob
        self.n_layers = args.n_layers
        self.hide_dim = args.hide_dim
        hide_dim = self.hide_dim
        self.alpha = args.alpha
        self.n_factors = args.n_factors
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.emb_dim = args.embed_size
        self.LayerNums = args.LayerNums
        # 特征融合系数
        self.wu1 = args.wu1
        self.wu2 = args.wu2
        self.wi1 = args.wi1
        self.wi2 = args.wi2
        # 融合系数
        self.ssl_ureg = args.ssl_ureg
        self.ssl_ireg = args.ssl_ireg
        self.ssl_temp_h = args.ssl_temp_h

        self.n_users_1 = config['userNum_S']
        self.n_users_2 = config['userNum_T']
        self.n_items_1 = config['itemNum_S']
        self.n_items_2 = config['itemNum_T']
        self.norm_adj_1 = config['adj_mat_S']
        self.norm_adj_2 = config['adj_mat_T']
        self.uuMat_1 = config['u2u_mat_S']
        self.uuMat_2 = config['u2u_mat_T']
        self.iiMat_1 = config['i2i_mat_S']
        self.iiMat_2 = config['i2i_mat_T']
        self.uiMat_1 = config['ui_mat_S']
        self.uiMat_2 = config['ui_mat_T']


        # GCN 编码层的初始化
        self.encoder = nn.ModuleList()  # 它用于存储一系列图卷积层
        for i in range(0, self.LayerNums):
            self.encoder.append(GCN_layer())  # 循环添加了多个 GCN_layer 到模型中。这些层将用于图数据的特征编码。

        # 两个线性变换层初始化，这些层用于执行从多个特征到隐藏表示的线性变换
        self.meta_netu = nn.Linear(self.hide_dim * 3, self.hide_dim, bias=True)
        self.meta_neti = nn.Linear(self.hide_dim * 3, self.hide_dim, bias=True)

        self.k = args.rank  # 低秩矩阵分解的维度
        k = self.k

        # MLP 层的初始化
        # 四个多层感知器（MLP）模型，每个 MLP 模型由两个隐藏层组成。这些模型将用于学习用户和物品的嵌入表示。
        self.mlp = MLP(hide_dim, hide_dim * k, hide_dim // 3, hide_dim * k)
        self.mlp1 = MLP(hide_dim, hide_dim * k, hide_dim // 3, hide_dim * k)
        self.mlp2 = MLP(hide_dim, hide_dim * k, hide_dim // 3, hide_dim * k)
        self.mlp3 = MLP(hide_dim, hide_dim * k, hide_dim // 3, hide_dim * k)

        # 一个神经网络模型的初始化部分，用于初始化用户和物品的嵌入（embedding）参数
        self.embedding_dict = nn.ParameterDict({
            'user_embedding_S': nn.Parameter(nn.init.normal_(t.empty(self.n_users_1, self.emb_dim), std=0.1)),
            'user_embedding_T': nn.Parameter(nn.init.normal_(t.empty(self.n_users_2, self.emb_dim), std=0.1)),
            'item_embedding_S': nn.Parameter(nn.init.normal_(t.empty(self.n_items_1, self.emb_dim), std=0.1)),
            'item_embedding_T': nn.Parameter(nn.init.normal_(t.empty(self.n_items_2, self.emb_dim), std=0.1))
        })

        # 首先，我们分配id-对应的嵌入e𝑢,ei
        self.gating_weightub = nn.Parameter(t.FloatTensor(1, hide_dim))  # 是一个可学习的模型参数，它是一个形状为 (1, hide_dim) 的权重矩阵
        nn.init.xavier_normal_(self.gating_weightub.data)  # 将其初始化为 Xavier 初始化的随机值
        self.gating_weightu = nn.Parameter(t.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weightu.data)
        self.gating_weightib = nn.Parameter(t.FloatTensor(1, hide_dim))
        nn.init.xavier_normal_(self.gating_weightib.data)
        self.gating_weighti = nn.Parameter(t.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weighti.data)

        # 创建一个 Sigmoid 激活函数实例，它将在模型的后续计算中使用。
        self.f = nn.Sigmoid()
        # 创建一个线性层实例，这个线性层的输入和输出维度都是 self.emb_dim。在模型的后续计算中，线性层会将输入数据进行线性变换，并输出到下一层。
        self.linear = nn.Linear(self.emb_dim, self.emb_dim)
        # 调用模型内部的 _init_graph 方法，用于初始化图结构。这部分代码很可能涉及到邻接矩阵的操作和初始化。
        self._init_graph()

    # 将稀疏矩阵转换为稀疏张量
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = t.Tensor(coo.row).long().to(self.device)
        col = t.Tensor(coo.col).long().to(self.device)
        index = t.stack([row, col])
        data = t.FloatTensor(coo.data).to(self.device)
        return t.sparse.FloatTensor(index, data, t.Size(coo.shape)).to(self.device)

    # 将矩阵A按照指定的份数（self.folds）进行切割，每份的长度由n_users和n_items决定。
    # 切割后的子矩阵被转换为稀疏张量，并存储在A_fold列表中返回
    def _split_A_hat(self, A, n_users, n_items):
        A_fold = []
        fold_len = (n_users + n_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = n_users + n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
        return A_fold

    # 用于初始化图结构。这部分代码很可能涉及到邻接矩阵的操作和初始化。
    def _init_graph(self):
        if self.A_split:
            self.Graph_1 = self._split_A_hat(self.norm_adj_1, self.n_users_1, self.n_items_1, self.device)
            self.Graph_2 = self._split_A_hat(self.norm_adj_2, self.n_users_2, self.n_items_2, self.device)
        else:
            self.Graph_1 = self._convert_sp_mat_to_sp_tensor(self.norm_adj_1)
            self.Graph_1 = self.Graph_1.coalesce().to(self.device)
            self.Graph_2 = self._convert_sp_mat_to_sp_tensor(self.norm_adj_2)
            self.Graph_2 = self.Graph_2.coalesce().to(self.device)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob, ori_graph):
        if self.A_split:
            graph = []
            for g in ori_graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(ori_graph, keep_prob)
        return graph

    # 该函数计算跨域推荐模型的嵌入向量    图数据、用户嵌入、物品嵌入、用户数量、物品数量
    def computer(self, graph, users_emb, items_emb, n_users, n_items):
        all_emb = t.cat([users_emb, items_emb])  # 将域中的用户和物品嵌入向量合并成一个大的向量 all_emb，并将其作为列表 embs 的初始元素。
        embs = [all_emb]

        if self.dropout:
            if self.training:
                print("droping")
                # 在代码中，当模型处于训练模式时，如果启用了 Dropout（self.dropout 为真），就会使用 keep_prob 参数来控制神经元的保留比例，从而执行 Dropout 操作。
                # 当模型不处于训练模式时，就不会应用 Dropout，即 g_droped 将直接使用原始的图数据。
                g_droped = self.__dropout(self.keep_prob, graph)
            else:
                g_droped = graph
        else:
            g_droped = graph

        # 在模型的每一层中进行图嵌入操作，生成多层的嵌入结果，并将这些结果存储在 layer_embs 列表中
        layer_embs = []  # 存储每层的嵌入结果
        # n_factors是一个指定的因子数量，n_layers==3 是神经网络的层数
        factor_num = [self.n_factors for i in range(self.n_layers)]  # 一个列表，包含了每层的因子数量
        for layer in range(self.n_layers):
            n_factors_l = factor_num[layer]  # 代表当前层的因子数量
            all_embs_tp = t.split(all_emb, int(self.emb_dim / n_factors_l), 1)  # 将 all_emb 沿着维度 1（列维度）分割成多个张量
            all_embs = []  # 用于存储当前层中的不同因子的嵌入结果
            for i in range(n_factors_l):  # 用于迭代当前层中的每个因子
                if self.A_split:
                    temp_emb = []
                    for f in range(len(g_droped)):
                        temp_emb.append(t.sparse.mm(g_droped[f], all_embs_tp[i]))
                    side_emb = t.cat(temp_emb, dim=0)  # 这个张量应该包含了当前层中当前因子的所有嵌入信息
                    all_embs.append(side_emb)
                else:
                    all_embs.append(t.sparse.mm(g_droped, all_embs_tp[i]))
            layer_embs.append(all_embs)  # 将当前层的嵌入结果列表 `all_embs` 添加到 `layer_embs` 列表中。  # 为了跟踪每一层的嵌入信息
            factor_embedding = t.cat([all_embs[0], all_embs[1]], dim=1)  # 将当前层中的不同因子的嵌入信息拼接在一起
            embs.append(factor_embedding)
            all_emb = factor_embedding
        # 将嵌入结果列表 embs 中的所有张量按照维度 1 进行堆叠，得到一个三维的张量，其中维度 0 表示样本数量，维度 1 表示图层数，维度 2 表示嵌入的维度。
        embs = t.stack(embs, dim=1)
        # 在维度 1 上计算所有图层嵌入的平均值，得到平均嵌入结果 light_out。这可以看作是对不同图层嵌入的汇总。
        light_out = t.mean(embs, dim=1)
        # 使用 torch.split 函数将平均嵌入 light_out 分割成用户嵌入向量 users 和物品嵌入向量 items，分割点为 [n_users, n_items]。
        users, items = t.split(light_out, [n_users, n_items])
        # 函数返回用户嵌入 users、物品嵌入 items，以及嵌入结果列表 layer_embs 的最后一项，表示最后一层的嵌入结果。
        return users, items, layer_embs[-1]

    # 自主门控机制
    def self_gatingu(self, em):
        return t.multiply(em, t.sigmoid(t.matmul(em, self.gating_weightu) + self.gating_weightub))
    def self_gatingi(self,em):
        return t.multiply(em, t.sigmoid(t.matmul(em,self.gating_weighti) + self.gating_weightib))

    # 邻居信息搭建  S 和 T
    def neighbor_information(self, uiMat, n_users):
        uimat = uiMat[: n_users, n_users:]
        # 将 uimat 转换为浮点数张量，首先使用 .tocoo() 方法将稀疏矩阵转换为坐标格式 (COO)，然后提取数据部分，并将其转换为 PyTorch 浮点数张量。
        values = t.FloatTensor(uimat.tocoo().data)  # 将uimat转换为浮点数张量
        # 获取稀疏矩阵中非零元素的行索引和列索引，然后将它们垂直堆叠在一起，形成一个 2xN 的 NumPy 数组，其中 N 是非零元素的数量。
        indices = np.vstack((uimat.tocoo().row, uimat.tocoo().col))
        i = t.LongTensor(indices)
        v = t.FloatTensor(values)
        shape = uimat.tocoo().shape
        # 这个稀疏张量表示了原始稀疏矩阵的部分，其非零元素和形状与原始矩阵相同。
        uimat1 = t.sparse.FloatTensor(i, v, t.Size(shape))
        uiadj = uimat1  # 这个张量表示用户到物品的连接关系
        iuadj = uimat1.transpose(0, 1)  # 它表示物品到用户的连接关系。通过将稀疏张量的维度进行转置，可以实现这一操作。
        return uiadj, iuadj

    # 执行元路径知识抽取和个性化转换  self.userEmbedding, self.ui_userEmbedding, self.itemEmbedding, self.ui_itemEmbedding
    def metafortansform(self, uiMat, n_users, auxiembedu, targetembedu, auxiembedi, targetembedi):
        uiadj, iuadj = self.neighbor_information(uiMat, n_users)

        # Neighbor information of the target node  这两行代码计算了用户和物品的邻居信息。
        uneighbor = t.matmul(uiadj.to(self.device), targetembedi)  # 包含了用户的物品邻居的信息
        ineighbor = t.matmul(iuadj.to(self.device), targetembedu)  # 包含了物品的用户邻居的信息

        # Meta-knowlege extraction  这两行代码使用元路径注意力模型来抽取用户和物品的元路径知识
        tembedu = ( self.meta_netu(t.cat((auxiembedu, targetembedu, uneighbor), dim=1).detach()))  # 附加的领域信息显式地增强了直接图连接的建模
        tembedi = (self.meta_neti(t.cat((auxiembedi, targetembedi, ineighbor), dim=1).detach()))

        """ Personalized transformation parameter matrix """
        # Low rank matrix decomposition
        metau1 = self.mlp(tembedu).reshape(-1, self.hide_dim, self.k)  # d*k  #  这些行执行了低秩矩阵分解，将元路径知识转换为权重矩阵。这些权重矩阵将用于个性化转换。
        metau2 = self.mlp1(tembedu).reshape(-1, self.k, self.hide_dim)  # k*d
        metai1 = self.mlp2(tembedi).reshape(-1, self.hide_dim, self.k)  # d*k
        metai2 = self.mlp3(tembedi).reshape(-1, self.k, self.hide_dim)  # k*d

        meta_biasu = (t.mean(metau1, dim=0))  # 这些行计算了元路径权重的偏置项，用于加权平均元路径知识。
        meta_biasu1 = (t.mean(metau2, dim=0))
        meta_biasi = (t.mean(metai1, dim=0))
        meta_biasi1 = (t.mean(metai2, dim=0))

        low_weightu1 = F.softmax(metau1 + meta_biasu, dim=1)  # 这些行使用 softmax 函数对元路径权重进行归一化，以确保它们的和为1
        low_weightu2 = F.softmax(metau2 + meta_biasu1, dim=1)
        low_weighti1 = F.softmax(metai1 + meta_biasi, dim=1)
        low_weighti2 = F.softmax(metai2 + meta_biasi1, dim=1)

        # The learned matrix as the weights of the transformed network
        # 这些行根据元路径权重将输入的用户和物品嵌入进行个性化转换。
        # tembedus 包含了转换后的用户嵌入，而 tembedis 包含了转换后的物品嵌入。这些个性化转换后的嵌入将用于更新用户和物品的嵌入。
        tembedus = (t.sum(t.multiply((auxiembedu).unsqueeze(-1), low_weightu1), dim=1))  # Equal to a two-layer linear network; Ciao and Yelp data sets are plus gelu activation function
        tembedus = t.sum(t.multiply((tembedus).unsqueeze(-1), low_weightu2), dim=1)
        tembedis = (t.sum(t.multiply((auxiembedi).unsqueeze(-1), low_weighti1), dim=1))
        tembedis = t.sum(t.multiply((tembedis).unsqueeze(-1), low_weighti2), dim=1)
        transfuEmbed = tembedus
        transfiEmbed = tembedis
        return transfuEmbed, transfiEmbed

    # 异构信息传播和聚合
    def information_dissemination_aggregation(self, n_users, n_items, ua_embeddings, ia_embeddings, uuMat, iiMat, uiMat, norm = 1):
        # HGCL
        user_index = np.arange(0, n_users)  # 创建一个物品源域索引数组
        item_index = np.arange(0, n_items)  # 创建一个用户源域索引数组
        # 创建一个包含用户索引和物品索引的数组。物品索引会被偏移 self.userNum，以与用户索引区分开
        ui_index = np.array(user_index.tolist() + [i + n_users for i in item_index])

        # Initialize Embeddings 初始化嵌入
        uu_embed0 = self.self_gatingu(ua_embeddings)  # 通过自主门控机制，得到用户到用户的嵌入 源域
        ii_embed0 = self.self_gatingi(ia_embeddings)
        self.ui_embed0 = t.cat([ua_embeddings, ia_embeddings], 0)  # 将用户嵌入和物品嵌入按行连接，表示用户和物品的联合嵌入
        self.all_user_embeddings = [uu_embed0]  # 初始化了用户和物品的嵌入列表，将初始嵌入添加到列表中。
        self.all_item_embeddings = [ii_embed0]
        self.all_ui_embeddings = [self.ui_embed0]

        # Encoder  用于处理每个 GCN 层, 异构消息传播
        for i in range(len(self.encoder)):
            layer = self.encoder[i]  # 从模型的 GCN 层列表中获取当前层 i 对应的 GCN 层
            if i == 0:
                userEmbeddings0 = layer(uu_embed0, uuMat, user_index)
                itemEmbeddings0 = layer(ii_embed0, iiMat, item_index)
                uiEmbeddings0 = layer(self.ui_embed0, uiMat, ui_index)
            else:
                # 如果不是第一层（即 else 分支），则使用前一层的嵌入作为输入，再次应用当前层的 GCN。
                userEmbeddings0 = layer(userEmbeddings, uuMat, user_index)
                itemEmbeddings0 = layer(itemEmbeddings, iiMat, item_index)
                uiEmbeddings0 = layer(uiEmbeddings, uiMat, ui_index)

            # 异构信息聚合
            # Aggregation of message features across the two related views in the middle layer then fed into the next layer
            # 将联合嵌入 uiEmbeddings0 分割成用户嵌入和物品嵌入，分割的方式是根据用户数和物品数进行分割
            self.ui_userEmbedding0, self.ui_itemEmbedding0 = t.split(uiEmbeddings0, [n_users, n_items])
            userEd = (userEmbeddings0 + self.ui_userEmbedding0) / 2.0  # 将当前层处理后的用户嵌入与联合嵌入的用户部分求平均
            itemEd = (itemEmbeddings0 + self.ui_itemEmbedding0) / 2.0

            userEmbeddings = userEd
            itemEmbeddings = itemEd
            uiEmbeddings = t.cat([userEd, itemEd], 0)  # 更新联合嵌入 uiEmbeddings，将其设置为合并后的用户嵌入和物品嵌入。

            if norm == 1:  # 选择是否对新嵌入进行 L2 归一化操作
                norm_embeddings = F.normalize(userEmbeddings0, p=2, dim=1)
                self.all_user_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(itemEmbeddings0, p=2, dim=1)
                self.all_item_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(uiEmbeddings0, p=2, dim=1)
                self.all_ui_embeddings += [norm_embeddings]
            else:
                self.all_user_embeddings += [userEmbeddings]
                self.all_item_embeddings += [itemEmbeddings0]
                self.all_ui_embeddings += [uiEmbeddings0]

        self.userEmbedding = t.stack(self.all_user_embeddings, dim=1)  # 每一列代表一个用户的嵌入
        self.userEmbedding = t.mean(self.userEmbedding, dim=1)  # 将所有用户的嵌入合并成一个均值向量，代表整体的用户嵌入。
        self.itemEmbedding = t.stack(self.all_item_embeddings, dim=1)  # 每一列代表一个物品的嵌入
        self.itemEmbedding = t.mean(self.itemEmbedding, dim=1)
        self.uiEmbedding = t.stack(self.all_ui_embeddings, dim=1)
        self.uiEmbedding = t.mean(self.uiEmbedding, dim=1)

        # 分割  提取用户和项目的元知识
        self.ui_userEmbedding, self.ui_itemEmbedding = t.split(self.uiEmbedding, [n_users, n_items])

        # Personalized Transformation of Auxiliary Domain Features
        # 该函数执行元路径注意力的变换操作，然后返回两个结果：metatsuembed 表示变换后的用户嵌入，metatsiembed 表示变换后的物品嵌入
        # 将执行元路径注意力变换后的用户和物品嵌入加入到原始嵌入中，以获得更新后的用户和物品嵌入。
        # ！！！最终嵌入！！！
        metatsuembed, metatsiembed = self.metafortansform(uiMat, n_users, self.userEmbedding, self.ui_userEmbedding, self.itemEmbedding, self.ui_itemEmbedding)

        # ！！！最终嵌入！！！
        self.userEmbedding = self.userEmbedding + metatsuembed
        self.itemEmbedding = self.itemEmbedding + metatsiembed


        return self.userEmbedding, self.itemEmbedding, self.ui_userEmbedding, self.ui_itemEmbedding

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse mattrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = t.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = t.from_numpy(sparse_mx.data).float()
        shape = t.Size(sparse_mx.shape)
        return t.sparse.FloatTensor(indices, values, shape)

    # 元路径正则化的计算过程
    # self.ui_userEmbedding[uid.cpu().numpy()],   (self.userEmbedding),   self.uuMat[uid.cpu().numpy()]
    def metaregular(self, em0, em, adj):
        # 用于对嵌入向量进行行和列的随机重排。这一步是为了生成负样本。
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[:, t.randperm(embedding.shape[1])]
            corrupted_embedding = corrupted_embedding[t.randperm(embedding.shape[0])]
            return corrupted_embedding

        # 用于计算两个向量之间的相似度分数，这里使用点积。
        def score(x1, x2):
            x1 = F.normalize(x1, p=2, dim=-1)
            x2 = F.normalize(x2, p=2, dim=-1)
            return t.sum(t.multiply(x1, x2), 1)

        user_embeddings = em

        # 计算邻接矩阵 adj 中每个节点的度（度是指与该节点相连的边的数量），并将结果转换为PyTorch张量
        Adj_Norm = t.from_numpy(np.sum(adj, axis=1)).float().to(self.device)
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)  # 将邻接矩阵 adj 转换为稀疏张量（PyTorch的稀疏矩阵表示）。
        edge_embeddings = t.spmm(adj.to(self.device), user_embeddings) / Adj_Norm  # 计算节点嵌入和边嵌入的乘积，然后除以节点度以获得归一化的边嵌入。
        user_embeddings = em0  # 这是为了生成正样本。
        graph = t.mean(edge_embeddings, 0)  # 计算所有边嵌入的平均值，得到一个全局的图嵌入。
        pos = score(user_embeddings, graph)  # 计算正样本的相似度分数，这里是 用户嵌入 与 全局图嵌入 的相似度。
        neg1 = score(row_column_shuffle(user_embeddings), graph)  # 生成负样本，首先对用户嵌入进行随机重排，然后计算与全局图嵌入的相似度。
        global_loss = t.mean(-t.log(t.sigmoid(pos - neg1)))  # 计算元路径正则化的全局损失，这里使用二元交叉熵 损失函数
        return global_loss

    # 定义计算 BPR（Bayesian Personalized Ranking）损失的函数，用于衡量模型的训练效果
    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = t.sum(t.mul(users, pos_items), axis=1)  # 通过用户嵌入向量和正样本物品嵌入向量的点积，计算出正样本的得分。
        neg_scores = t.sum(t.mul(users, neg_items), axis=1)  # 通过用户嵌入向量和负样本物品嵌入向量的点积，计算出负样本的得分。

        # 计算基于 BPR 的 MF（Matrix Factorization）损失。这里使用了 softplus 函数，它可以将任意输入映射到非负值，以确保损失始终为正。
        mf_loss = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))
        # mf_loss = t.mean(-t.log(t.sigmoid(pos_scores - neg_scores)))  # 二元交叉熵损失函数
        return mf_loss

# 不按照批次[uid.cpu().numpy()]计算元路径正则化损失
    def calculate_metaregloss(self, ui_userEmbedding, userEmbedding, ui_itemEmbedding, itemEmbedding, uuMat, iiMat):
        # Regularization: the constraint of transformed reasonableness
        self.reg_lossu = self.metaregular(ui_userEmbedding, userEmbedding, uuMat)
        self.reg_lossi = self.metaregular(ui_itemEmbedding, itemEmbedding, iiMat)
        metaregloss = (self.reg_lossu + self.reg_lossi) / 2.0 # 元路径正则化损失

        return metaregloss

    # 这段代码的目的是计算 SSL（Semi-Supervised Learning）损失，用于训练中的半监督学习。
    # SSL损失可以帮助在跨域推荐中学习不同域之间的共享信息和特定信息
    def calc_ssl_loss_strategy(self, layer_embed_1, layer_embed_2, n_users_1, n_users_2, n_items_1, n_items_2):
        # # 分割   S
        # users_1, items_1 = t.split(layer_embed_1, [n_users_1, n_items_1])
        # users_up_1, items_up_1, _, _ = self.information_dissemination_aggregation(n_users_1, n_items_1, users_1, items_1, self.uuMat_1, self.iiMat_1, self.uiMat_1)
        # # 组合   S
        # final_layer_embed_1 = t.cat([users_up_1, items_up_1])
        #
        # # 分割   T
        # users_2, items_2 = t.split(layer_embed_2, [n_users_2, n_items_2])
        # users_up_2, items_up_2, _, _ = self.information_dissemination_aggregation(n_users_2, n_items_2, users_2, items_2, self.uuMat_2, self.iiMat_2, self.uiMat_2)
        # # 组合   T
        # final_layer_embed_2 = t.cat([users_up_2, items_up_2])

        # 提取不变和特定信息的嵌入向量
        invariant_embed_1, specific_embed_1 = layer_embed_1[0], layer_embed_1[1]
        # 从中提取用户的部分
        invariant_u_embed_1, specific_u_embed_1 = invariant_embed_1[:n_users_1], specific_embed_1[:n_users_1]

        invariant_embed_2, specific_embed_2 = layer_embed_2[0], layer_embed_2[1]
        invariant_u_embed_2, specific_u_embed_2 = invariant_embed_2[:n_users_2], specific_embed_2[:n_users_2]

        # 向量标准化， 对提取的不变信息  用户嵌入向量  进行L2 归一化，确保向量的范数为1
        normalize_invariant_user_1 = t.nn.functional.normalize(invariant_u_embed_1, p=2, dim=1)
        normalize_invariant_user_2 = t.nn.functional.normalize(invariant_u_embed_2, p=2, dim=1)

        normalize_specific_user_1 = t.nn.functional.normalize(specific_u_embed_1, p=2, dim=1)
        normalize_specific_user_2 = t.nn.functional.normalize(specific_u_embed_2, p=2, dim=1)

        # 计算两个不同域的不变信息用户嵌入向量的点积，作为正样本的得分。正样本代表的是不同域之间共享信息的相似性。
        pos_score_user = t.sum(t.mul(normalize_invariant_user_1, normalize_invariant_user_2), dim=1)

        # 计算不同组合的不变信息和特定信息用户嵌入向量的点积，作为负样本的得分。特定信息和不同域之间的差异性
        # 标准的余弦相似度代码：cosine_similarity = torch.sum(normalize_invariant_user_1 * normalize_specific_user_1, dim=1)
        neg_score_1 = t.sum(t.mul(normalize_invariant_user_1, normalize_specific_user_1), dim=1)
        neg_score_2 = t.sum(t.mul(normalize_invariant_user_2, normalize_specific_user_2), dim=1)
        neg_score_3 = t.sum(t.mul(normalize_specific_user_1, normalize_specific_user_2), dim=1)

        # 为什么不计算
        # neg_score_5 = torch.sum(torch.mul(normalize_invariant_user_2, normalize_specific_user_1), dim=1)

        # 使用矩阵乘法计算两个不变信息用户嵌入向量之间的点积，作为额外的负样本得分。
        # 帮助模型更好地区分不同域之间的用户关系
        neg_score_4 = t.matmul(normalize_invariant_user_2, normalize_specific_user_1.T)

        # 对得分进行指数化，以便进行后续的损失计算。
        # self.ssl_temp 控制了损失函数中指数化得分的温度，影响了得分之间的相对大小。
        # 温度参数是用来调整 softmax 操作的输出，使其更平滑或更尖锐。
        # 在这个上下文中，self.ssl_temp 的值为 0.1，意味着在计算 SSL 损失时将使用较低的温度。
        pos_score = t.exp(pos_score_user / self.ssl_temp)
        neg_score_1 = t.exp(neg_score_1 / self.ssl_temp)
        neg_score_2 = t.exp(neg_score_2 / self.ssl_temp)
        neg_score_3 = t.exp(neg_score_3 / self.ssl_temp)
        neg_score_4 = t.sum(t.exp(neg_score_4 / self.ssl_temp), dim=1)

        # 最小化正样本得分与所有负样本得分的比值的负对数，以便提高正样本得分并降低负样本得分，从而改善模型的区分性能
        ssl_loss_user = -t.sum(t.log(pos_score / (neg_score_1 + neg_score_2 + neg_score_3 + pos_score +
                                                          neg_score_4)))
        # SSL 损失和温度参数一起，可以帮助模型更好地学习 跨域推荐中用户之间的共享信息和特定信息
        ssl_loss = ssl_loss_user
        return ssl_loss

    # Contrastive Learning
    def ssl_loss(self, data1, data2, index):
        # 是一个包含索引的张量，它用于指示应该在输入数据中选择哪些样本进行计算
        index = t.from_numpy(index)
        index = t.unique(index)  # 首先确保索引 index 中的值不重复，这可以避免重复计算相同的样本对
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        pos_score = t.sum(t.mul(norm_embeddings1, norm_embeddings2), dim=1)  # 正样本对的相似度分数，这里采用了向量点积来度量它们之间的相似性。
        all_score = t.mm(norm_embeddings1, norm_embeddings2.T)  # 所有样本对的相似度分数矩阵，其中的每个元素表示一个样本对的相似性。
        pos_score = t.exp(pos_score / self.ssl_temp_h)  # 应用了 softmax 操作，这有助于将分数转化为概率分布
        all_score = t.sum(t.exp(all_score / self.ssl_temp_h), dim=1)
        # 目标是使正样本对的相似度分数更大，而将其他样本对的相似度分数尽可能地降低
        ssl_loss = (-t.sum(t.log(pos_score / ((all_score)))) / (len(index)))
        return ssl_loss

    def predictModel(user, pos_i, neg_j, isTest=False):
        if isTest:
            pred_pos = t.sum(user * pos_i, dim=1)
            return pred_pos
        else:
            pred_pos = t.sum(user * pos_i, dim=1)
            pred_neg = t.sum(user * neg_j, dim=1)
            return pred_pos, pred_neg

    # 前向传播函数，分别是：用户-项目交互数据中的用户索引（用户ID）、用户-项目交互数据中的项目索引（正样本）、随机项目索引（负样本）
    def forward(self, users_1, pos_items_1, neg_items_1, users_2, pos_items_2, neg_items_2, userindex_S, itemindex_S, userindex_T, itemindex_T, train):
        # 生成嵌入向量
        self.ua_embeddings_1, self.ia_embeddings_1, self.layer_embeddings_1 = self.computer(self.Graph_1, self.embedding_dict['user_embedding_S'],
                                                                         self.embedding_dict['item_embedding_S'], self.n_users_1, self.n_items_1)
        self.ua_embeddings_2, self.ia_embeddings_2, self.layer_embeddings_2 = self.computer(self.Graph_2, self.embedding_dict['user_embedding_T'],
                                                                         self.embedding_dict['item_embedding_T'], self.n_users_2, self.n_items_2)

        # 融合了异构信息之后得到的用户、物品、layer嵌入向量 uu ii ui_u ui_i
        self.userEmbedding_1, self.itemEmbedding_1, self.ui_userEmbedding_1, self.ui_itemEmbedding_1 = self.information_dissemination_aggregation(self.n_users_1,
                                                                                                self.n_items_1,
                                                                                                self.ua_embeddings_1,
                                                                                                self.ia_embeddings_1,
                                                                                                self.uuMat_1,
                                                                                                self.iiMat_1,
                                                                                                self.uiMat_1)
        self.userEmbedding_2, self.itemEmbedding_2, self.ui_userEmbedding_2, self.ui_itemEmbedding_2  = self.information_dissemination_aggregation(self.n_users_2,
                                                                                                self.n_items_2,
                                                                                                self.ua_embeddings_2,
                                                                                                self.ia_embeddings_2,
                                                                                                self.uuMat_2,
                                                                                                self.iiMat_2,
                                                                                                self.uiMat_2)

        self.fu1 = self.wu1 * self.ui_userEmbedding_1 + self.wu2 * self.userEmbedding_1
        self.fi1 = self.wi1 * self.ui_itemEmbedding_1 + self.wi2 * self.itemEmbedding_1
        self.fu2 = self.wu1 * self.ui_userEmbedding_2 + self.wu2 * self.userEmbedding_2
        self.fi2 = self.wi1 * self.ui_itemEmbedding_2 + self.wi2 * self.itemEmbedding_2

        # 根据索引从不同的嵌入矩阵中提取出对应的用户和正样本和负样本的嵌入向量，以及预训练的......
        self.u_g_embeddings_1 = t.index_select(self.fu1, 0, t.LongTensor(users_1).to(self.device))
        self.pos_i_g_embeddings_1 = t.index_select(self.fi1, 0, t.LongTensor(pos_items_1).to(self.device))
        self.neg_i_g_embeddings_1 = t.index_select(self.fi1, 0, t.LongTensor(neg_items_1).to(self.device))
        # self.u_g_embeddings_pre_1 = t.index_select(self.embedding_dict['user_embedding_S'], 0, t.LongTensor(users_1).to(self.device))
        # self.pos_i_g_embeddings_pre_1 = t.index_select(self.embedding_dict['item_embedding_S'], 0, t.LongTensor(pos_items_1).to(self.device))
        # self.neg_i_g_embeddings_pre_1 = t.index_select(self.embedding_dict['item_embedding_S'], 0, t.LongTensor(neg_items_1).to(self.device))

        self.u_g_embeddings_2 = t.index_select(self.fu2, 0, t.LongTensor(users_2).to(self.device))
        self.pos_i_g_embeddings_2 = t.index_select(self.fi2, 0, t.LongTensor(pos_items_2).to(self.device))
        self.neg_i_g_embeddings_2 = t.index_select(self.fi2, 0, t.LongTensor(neg_items_2).to(self.device))
        # self.u_g_embeddings_pre_2 = t.index_select(self.embedding_dict['user_embedding_T'], 0, t.LongTensor(users_2).to(self.device))
        # self.pos_i_g_embeddings_pre_2 = t.index_select(self.embedding_dict['item_embedding_T'], 0, t.LongTensor(pos_items_2).to(self.device))
        # self.neg_i_g_embeddings_pre_2 = t.index_select(self.embedding_dict['item_embedding_T'], 0, t.LongTensor(neg_items_2).to(self.device))

        metaregloss_1 = 0  # 元路径正则化损失
        metaregloss_2 = 0
        variationloss = 0 # dccdr LOSS
        ssl_loss_T = 0
        ssl_loss_S = 0
        regLoss_S = 0
        regLoss_T = 0

        if train == True:  # 只有在训练模式下才会计算元路径正则化损失。
            # # 计算bpr损失
            mf_loss_1 = self.create_bpr_loss(self.u_g_embeddings_1, self.pos_i_g_embeddings_1, self.neg_i_g_embeddings_1)
            mf_loss_2 = self.create_bpr_loss(self.u_g_embeddings_2, self.pos_i_g_embeddings_2, self.neg_i_g_embeddings_2)
            # 计算SSL损失
            ssl_loss = self.calc_ssl_loss_strategy(self.layer_embeddings_1, self.layer_embeddings_2, self.n_users_1, self.n_users_2, self.n_items_1, self.n_items_2)
            # 基于BPR的 MF（Matrix Factorization）损失 + 嵌入项的正则化损失 ：SSL损失函数（额外乘以一个权重常常用来平衡不同损失项的贡献）
            variationloss = mf_loss_1 + mf_loss_2 + self.alpha * ssl_loss
            # Regularization: the constraint of transformed reasonableness

            metaregloss_1 = self.calculate_metaregloss(self.ui_userEmbedding_1[userindex_S.cpu().numpy()], self.userEmbedding_1, self.ui_itemEmbedding_1[itemindex_S.cpu().numpy()],
                                                       self.itemEmbedding_1, self.uuMat_1[userindex_S.cpu().numpy()], self.iiMat_1[itemindex_S.cpu().numpy()])
            metaregloss_2 = self.calculate_metaregloss(self.ui_userEmbedding_2[userindex_T.cpu().numpy()], self.userEmbedding_2, self.ui_itemEmbedding_2[itemindex_T.cpu().numpy()],
                                                       self.itemEmbedding_2, self.uuMat_2[userindex_T.cpu().numpy()], self.iiMat_2[itemindex_T.cpu().numpy()])
            # Contrastive Learning of collaborative relations
            ssl_loss_user_S = self.ssl_loss(self.ui_userEmbedding_1, self.userEmbedding_1, users_1)
            ssl_loss_user_T = self.ssl_loss(self.ui_userEmbedding_2, self.userEmbedding_2, users_2)
            ssl_loss_item_S = self.ssl_loss(self.ui_itemEmbedding_1, self.itemEmbedding_1, pos_items_1)
            ssl_loss_item_T = self.ssl_loss(self.ui_itemEmbedding_2, self.itemEmbedding_2, pos_items_2)
            ssl_loss_S = self.ssl_ureg * ssl_loss_user_S + self.ssl_ireg * ssl_loss_item_S
            ssl_loss_T = self.ssl_ureg * ssl_loss_user_T + self.ssl_ireg * ssl_loss_item_T

            # 计算regloss
            regLoss_S = (t.norm(self.fu1[users_1]) ** 2 + t.norm(self.fi1[pos_items_1]) ** 2 + t.norm(
                self.fi1[neg_items_1]) ** 2)
            regLoss_T = (t.norm(self.fu2[users_1]) ** 2 + t.norm(self.fi2[pos_items_1]) ** 2 + t.norm(
                self.fi2[neg_items_1]) ** 2)

        trainneed = {}
        trainneed['metaregloss_1'] = metaregloss_1
        trainneed['metaregloss_2'] = metaregloss_2
        trainneed['ssl_loss_S'] = ssl_loss_S
        trainneed['ssl_loss_T'] = ssl_loss_T
        trainneed['variationloss'] = variationloss
        trainneed['regLoss_S'] = regLoss_S
        trainneed['regLoss_T'] = regLoss_T
        return trainneed


    def get_embeddings(self, users_1, pos_items_1, neg_items_1, users_2, pos_items_2, neg_items_2):
        self.u_g_embeddings_1 = t.index_select(self.fu1, 0, t.LongTensor(users_1).to(self.device))
        self.pos_i_g_embeddings_1 = t.index_select(self.fi1, 0, t.LongTensor(pos_items_1).to(self.device))
        self.neg_i_g_embeddings_1 = t.index_select(self.fi1, 0, t.LongTensor(neg_items_1).to(self.device))

        self.u_g_embeddings_2 = t.index_select(self.fu2, 0, t.LongTensor(users_2).to(self.device))
        self.pos_i_g_embeddings_2 = t.index_select(self.fi2, 0, t.LongTensor(pos_items_2).to(self.device))
        self.neg_i_g_embeddings_2 = t.index_select(self.fi2, 0, t.LongTensor(neg_items_2).to(self.device))

class GCN_layer(nn.Module):
    def __init__(self):
        super(GCN_layer, self).__init__()

    # 这个方法用于将一个 scipy 稀疏矩阵转换为 PyTorch 的稀疏张量
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = t.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = t.from_numpy(sparse_mx.data).float()
        shape = t.Size(sparse_mx.shape)
        return t.sparse.FloatTensor(indices, values, shape)

    # 这个方法用于对输入的邻接矩阵进行对称归一化
    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        rowsum = np.maximum(rowsum, 1e-12)
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()

    # 这个方法是 GCN_layer 的前向传播函数，用于对输入的特征数据进行 GCN 卷积操作
    # 它接受图的 邻接矩阵 和 节点特征 ，并根据 邻接关系 进行卷积操作，然后更新指定节点的特征。
    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = self.normalize_adj(subset_Mat)  # 将邻接矩阵对称归一化
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).cuda()   # # 这个要记得改成cuda

        start_time = timeit.default_timer()
        out_features = t.spmm(subset_sparse_tensor, subset_features).cuda()
        end_time = timeit.default_timer()
        # print("class GCN_layer in t.spmm:", end_time - start_time)

        new_features = t.empty(features.shape).cuda()    # # 这个要记得改成cuda
        new_features[index] = out_features
        dif_index = np.setdiff1d(t.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]
        return new_features


class MLP(t.nn.Module):
    # input_dim：输入特征的维度。
    # feature_dim：中间特征的维度（如果 feature_pre 为 True，则表示中间层的维度）。
    # hidden_dim：中间层的维度（如果 feature_pre 为 False，则表示中间层的维度）。
    # output_dim：输出层的维度。
    # feature_pre：一个布尔值，表示是否在第一层应用线性变换到 feature_dim 维度。
    # layer_num：MLP 的层数，默认为 2。
    # dropout：一个布尔值，表示是否在中间层应用 dropout
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre =   nn.Linear(input_dim, feature_dim,bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(feature_dim, output_dim,bias=True)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu=nn.PReLU().cuda()   # # 这个要记得改成cuda
        x = prelu(x)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x