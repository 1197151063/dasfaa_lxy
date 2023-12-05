import json
import random
import timeit
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.sparse import dok_matrix
from tqdm import tqdm
import scipy.sparse as sp

def get_distance_mat(mat_path, turnnum_path, userNum, itemNum, ratings_train):

    try:
        i2i_mat = sp.load_npz(mat_path + 'i2i_mat.npz')
        u2u_mat = sp.load_npz(mat_path + 'u2u_mat.npz')
        ui_mat = sp.load_npz(mat_path + 'ui_mat.npz')
        print('already i2i_mat:', i2i_mat.shape, 'already u2u_mat:', u2u_mat.shape, 'already ui_mat:', ui_mat.shape)
    except Exception:
        # i2i_mat = create_i2i_mat_various(mat_path, turnnum_path, itemNum)  # 保留项目类别多样性
        i2i_mat = create_i2i_mat_main(mat_path, turnnum_path, itemNum)  # 选择主要类别
        u2u_mat = create_u2u_mat(mat_path, turnnum_path, userNum)
        ui_mat = create_ui_mat(mat_path, userNum, itemNum, ratings_train)
    return i2i_mat, u2u_mat, ui_mat

def create_u2u_mat(mat_path, load_path, userNum):
    # u2u_mat
    trust_info_set = load_trust_list(load_path, userNum)
    # 创建一个稀疏张量 trustMat
    trustMat = sp.dok_matrix((userNum, userNum))
    # 遍历trust集合中的每一对记录
    for trusterID, trusteeID in tqdm(trust_info_set, desc='u2u_mat:', unit='records'):
        # 将(trusterID, trusteeID)添加到trustMat中
        trustMat[trusterID, trusteeID] = 1
    u2u_mat = sp.dok_matrix((userNum, userNum))
    tmp_trustMat = trustMat.tocoo()
    uidList1, uidList2 = tmp_trustMat.row, tmp_trustMat.col
    u2u_mat[uidList1, uidList2] = 1.0
    u2u_mat[uidList2, uidList1] = 1.0
    u2u_mat = (u2u_mat + sp.eye(userNum)).tocsr()   # 并将其与单位矩阵相加，以确保对角线上的值为1.0，表示用户信任自己。
    sp.save_npz(mat_path + 'u2u_mat.npz', u2u_mat)
    print("!!! u2u_mat is done !!!")
    return u2u_mat

def load_trust_list(turnnum_path, userNum):
    with open(turnnum_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 创建一个字典来存储userID对应的trust总和
    user_trust_sum = {}
    # 遍历数据
    for record in tqdm(data, desc='user_trust_sum:', unit='records'):
        userID = record['userID']
        trust = record['trust']
        # 如果trust不为0，将其累加到user_trust_sum中
        if trust != 0:
            if userID in user_trust_sum:
                user_trust_sum[userID] += trust
            else:
                user_trust_sum[userID] = trust
        user_trust_sum[userID] = trust

    # 将结果存储为一个 n*2 的矩阵
    trustMat = set()
    for userID, trust in tqdm(user_trust_sum.items(), desc='trustMat:', unit='records'):
        trustMat.add((int(userID), int(trust)))

    # 创建一个空集合来存储多对一的信息
    trust_info_set = set()
    # 遍历trustMat中的每条记录
    for user_id, truster_count in tqdm(trustMat, desc='trust_info_set:', unit='records'):
        # 从用户总数中随机选择信任者数量个userID作为信任者ID
        remaining_users = set(range(userNum)) - {user_id}
        trusters = random.sample(remaining_users, truster_count)
        # 将信任者ID和用户ID组成多对一的信息，并添加到集合中
        for trustee in trusters:
            trust_info_set.add((trustee, user_id))
    return trust_info_set

def create_ui_mat(mat_path, userNum, itemNum, ratings_train):
    ...
    # 2.将ratings_train转换成ratingmat
    # 3.构造u-imat
    ratingMat = sp.dok_matrix((userNum, itemNum))
    for uid in tqdm(ratings_train.keys(), desc='to ratingMat:', unit='records'):
        for iid in ratings_train[uid]:
                ratingMat[uid, iid] = 1
    ui_mat = sp.dok_matrix((userNum+itemNum, userNum+itemNum))
    ratingMat_T = ratingMat.T
    for i in tqdm(range(userNum + itemNum), desc='ui_mat:', unit='records'):
        if i < userNum:
            ui_mat[i, userNum:] = ratingMat[i]
        else:
            ui_mat[i, :userNum] = ratingMat_T[i - userNum]
    ui_mat = ui_mat.tocsr()
    sp.save_npz(mat_path + 'ui_mat.npz', ui_mat)
    print("!!! ui_mat is done !!!")
    return ui_mat


# 1、保留多样性 various
def create_i2i_mat_various(mat_path, turnnum_path, itemNum):
    # 1. 构造categoryMat
    with open(turnnum_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    seen_itemIDs = set()
    itemCategoryDict = {}
    for record in tqdm(data, desc='categoryMat_categoryDict:', unit='records'):
        itemID = record["itemID"]
        category = record["category"]
        # 检查是否已存在相同的itemID，如果不存在则赋值
        if (itemID, category) not in seen_itemIDs:
            seen_itemIDs.add((itemID, category))
            # 总结一个itemID所属的所有类别，构建itemCategoryDict
            if itemID in itemCategoryDict:
                itemCategoryDict[itemID].append(category)
            else:
                itemCategoryDict[itemID] = [category]
    # 3. 为每个项目的每个类别分别构建距离矩阵
    # 获取所有类别
    all_categories = sorted(set(category for categories in itemCategoryDict.values() for category in categories))
    ItemDistance_mat = np.zeros((itemNum, itemNum, len(all_categories)))
    # 构建距离矩阵
    for i in tqdm(range(itemNum), desc='i2i_mat:', unit='records'):
        categories_i = itemCategoryDict[i]
        for j in range(itemNum):
            categories_j = itemCategoryDict[j]
            # 对每个类别构建距离矩阵
            for k, category in enumerate(all_categories):
                if category in categories_i and category in categories_j:
                    ItemDistance_mat[i, j, k] = 2.0
                else:
                    ItemDistance_mat[i, j, k] = 0.0

    # 将多个距离矩阵组合在一起
    # 合并多维矩阵的第三层
    i2i_mat = np.sum(ItemDistance_mat, axis=2).reshape((itemNum, itemNum))
    i2i_mat_spdok = dok_matrix(i2i_mat)
    i2i_mat_csr = (i2i_mat_spdok + sp.eye(itemNum)).tocsr()
    sp.save_npz(mat_path + 'i2i_mat.npz', i2i_mat_csr)
    print("!!! i2i_mat is done !!!")
    return i2i_mat_csr

# 2. 选择主要类别
def create_i2i_mat_main(mat_path, turnnum_path, itemNum):
    # 1. 构造categoryMat
    with open(turnnum_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    seen_itemIDs = set()
    itemCategoryDict = {}
    for record in tqdm(data, desc='categoryMat_categoryDict:', unit='records'):
        itemID = record["itemID"]
        category = record["category"]
        # 检查是否已存在相同的itemID，如果不存在则赋值
        if (itemID, category) not in seen_itemIDs:
            seen_itemIDs.add((itemID, category))
            # 总结一个itemID所属的所有类别，构建itemCategoryDict
            if itemID in itemCategoryDict:
                itemCategoryDict[itemID].append(category)
            else:
                itemCategoryDict[itemID] = [category]

    # 选择主要类别
    categoryMat = sp.dok_matrix((itemNum, 1))
    for i, (itemID, categories) in enumerate(itemCategoryDict.items()):
        main_category = max(set(categories), key=categories.count)
        categoryMat[i, 0] = main_category
    print("!!! categoryMat is done !!!")

    categoryDict = {}  # 用于存储每个类别对应的物品列表
    categoryData = categoryMat.toarray().reshape(-1)
    for i in range(categoryData.size):
        iid = i  # 物品ID
        typeid = categoryData[i]  # 类别ID
        if typeid in categoryDict:
            categoryDict[typeid].append(iid)
        else:
            categoryDict[typeid] = [iid]
    print("!!! categoryDict is done !!!")

    i2i_mat = sp.dok_matrix((itemNum, itemNum))
    for i in tqdm(range(itemNum), desc='i2i_mat:', unit='records'):
        itemType = categoryMat[i, 0]
        itemList = categoryDict[itemType]
        itemList = np.array(itemList)
        # 进行项目选取值判断
        if itemList.size < 200:
            proportion = 1
        else:
            proportion = 0.005
        itemList2 = np.random.choice(itemList, size=int(itemList.size * proportion), replace=False)
        itemList2 = itemList2.tolist()
        tmp = [i] * len(itemList2)
        i2i_mat[tmp, itemList2] = 2.0
        i2i_mat[itemList2, tmp] = 2.0

    ##final result
    start_time = timeit.default_timer()
    i2i_mat = (i2i_mat + sp.eye(itemNum)).tocsr()
    end_time = timeit.default_timer()
    print("tocsr():", end_time - start_time)

    start_time = timeit.default_timer()
    sp.save_npz(mat_path + 'i2i_mat.npz', i2i_mat)
    end_time = timeit.default_timer()
    print("sp.save_npz:", end_time - start_time)

    print("!!! i2i_mat is done !!!")
    return i2i_mat