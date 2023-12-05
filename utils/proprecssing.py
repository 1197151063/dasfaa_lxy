import json
from collections import defaultdict
from tqdm import tqdm


def pre_data(pre_path_S, pre_path_T, pre_save_path_S, pre_save_path_T, score, num):
    data_list_S = []
    # 1. 打开JSON文件并加载数据
    with open(pre_path_S, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_list_S.append(data)

    data_list_T = []
    with open(pre_path_T, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_list_T.append(data)

    # 3.筛选出评分大于等于4，且至少有5个物品交互的用户记录，pre_data
    pre_S = pre_processing(data_list_S, score, num)
    pre_T = pre_processing(data_list_T, score, num)

    with open(pre_save_path_S, 'w') as of:
        json.dump(pre_S, of, indent=4)
    print("!!! pre_S is done !!!")
    with open(pre_save_path_T, 'w') as of:
        json.dump(pre_T, of, indent=4)
    print("!!! pre_T is done !!!")

def pre_processing(data, score, num):
    # 1. 选出overall大于等于4的所有记录
    fit_rating = [record for record in data if record["overall"] >= score]
    # 2. user-item原始交互列表
    # 创建一个字典来存储用户到项目的一对多索引表
    user_item_index = {}
    # 遍历新的数据列表
    for record in tqdm(fit_rating, desc='interact data:', unit=' records'):
        user_number = record['reviewerID']
        item_number = record['asin']

        # 检查用户是否在字典中
        if user_number in user_item_index:
            # 如果用户在字典中，将项目编号添加到用户的项目列表中
            user_item_index[user_number].append(item_number)
        else:
            # 如果用户不在字典中，创建一个新的项目列表
            user_item_index[user_number] = [item_number]
    # 3.计算itemID值的长度，如果小于5，则记录对应的userID的值
    useful_users = set()
    for user_number, item_number in tqdm(user_item_index.items(), desc='save useful reviewerID:', unit=' records'):
        if len(item_number) >= num:
            useful_users.add(user_number)
    # 4.移除不满足条件的用户
    new_fit_rating = []  # 创建一个新的列表，用于存储不需要删除的记录
    for record in tqdm(fit_rating, desc='useful record:', unit=' records'):
        userID = record['reviewerID']
        # 如果userID不在users_to_remove列表中，将记录添加到新的列表中
        if userID in useful_users:
            new_fit_rating.append(record)
    print("!!! new_fit_rating is done !!!")
    return new_fit_rating

# 得到项目一对多的字典
def item_list(pre_save_path_S, pre_save_path_T):

    # 1. 打开JSON文件并加载数据
    with open(pre_save_path_S, 'r', encoding='utf-8') as f:
        data_list_S = json.load(f)

    with open(pre_save_path_T, 'r', encoding='utf-8') as f:
        data_list_T = json.load(f)

    user_item_index1 = {}
    for record in tqdm(data_list_S, desc='interact data S:', unit=' records'):
        user_number = record['reviewerID']
        item_number = record['asin']

        # 检查用户是否在字典中
        if user_number in user_item_index1:
            # 如果用户在字典中，将项目编号添加到用户的项目列表中
            user_item_index1[user_number].append(item_number)
        else:
            # 如果用户不在字典中，创建一个新的项目列表
            user_item_index1[user_number] = [item_number]
    print("!!! user_item_index_S is done !!!")

    user_item_index2 = {}
    for record in tqdm(data_list_T, desc='interact data T:', unit=' records'):
        user_number = record['reviewerID']
        item_number = record['asin']

        # 检查用户是否在字典中
        if user_number in user_item_index2:
            # 如果用户在字典中，将项目编号添加到用户的项目列表中
            user_item_index2[user_number].append(item_number)
        else:
            # 如果用户不在字典中，创建一个新的项目列表
            user_item_index2[user_number] = [item_number]
    print("!!! user_item_index_T is done !!!")
    return user_item_index1, user_item_index2

# 创造项目索引map，控制项目数量到10000个
def item_id_list(dict1, dict2):
    item_cnt_map1, item_cnt_map2 = defaultdict(int), defaultdict(int)
    for k, v in dict1.items():
        for item in v:
            item_cnt_map1[item] += 1
    for k, v in dict2.items():
        for item in v:
            item_cnt_map2[item] += 1
    #选取排名前10000的项目ID
    item_cnt_map1 = sorted(item_cnt_map1.items(), key=lambda d: d[1], reverse=True)[:10000]
    item_cnt_map2 = sorted(item_cnt_map2.items(), key=lambda d: d[1], reverse=True)[:10000]

    item_id_map1, item_id_map2 = defaultdict(int), defaultdict(int)
    item_id_1, item_id_2 = 0, 0
    for i, j in tqdm(item_cnt_map1, desc='item_cnt_map1 data:', unit=' records'):
        item_id_map1[i] = item_id_1
        item_id_1 += 1
    print("!!! item_id_map_S is done !!!")
    for i, j in tqdm(item_cnt_map2, desc='item_cnt_map2 data:', unit=' records'):
        item_id_map2[i] = item_id_2
        item_id_2 += 1
    print("!!! item_id_map_T is done !!!")
    return item_id_map1, item_id_map2



# 得到最终的要处理的列表数据
# 同时创建用户索引
def overlap_data(item_id_map1, item_id_map2, dict1, dict2, num, filepath):
    set_1 = set(dict1.keys())
    set_2 = set(dict2.keys())
    user_id = 0
    intersect_set = set_1 & set_2
    distinct_set_1 = set_1 - intersect_set
    distinct_set_2 = set_2 - intersect_set
    user_id_mapping_common = {}
    user_id_mapping_distinct1 = {}
    user_id_mapping_distinct2 = {}
    for u in tqdm(intersect_set, desc='common dat:', unit=' records'):
        item_list_1, item_list_2 = dict1[u], dict2[u]   # 项目列表
        added_items_1, added_items_2 = set(), set()
        for item in item_list_1:
            if item in item_id_map1.keys():
                # item_li_1.append(item_id_map1[item])
                added_items_1.add(item_id_map1[item])
        for item in item_list_2:
            if item in item_id_map2.keys():
                # item_li_2.append(item_id_map2[item])
                added_items_2.add(item_id_map2[item])
        item_li_1 = list(added_items_1)
        item_li_2 = list(added_items_2)
        if len(item_li_1) >= num and len(item_li_2) >= num:
            with open(filepath + '/common_S.dat', 'a') as fout1:
                fout1.write(str(user_id))
                for t in item_li_1:
                    fout1.write(' '+str(t))
                fout1.write('\n')
            with open(filepath + '/common_T.dat', 'a') as fout2:
                fout2.write(str(user_id))
                for t in item_li_2:
                    fout2.write(' '+str(t))
                fout2.write('\n')
            user_id_mapping_common[u] = user_id  # 存储原userID和新的编号的对应关系
            user_id += 1
    print("common_user: {}".format(user_id), "!!! common_dat are done !!!")

    common_users = user_id
    for u in tqdm(distinct_set_1, desc='distinct_S dat:', unit=' records'):
        item_list_1 = dict1[u]
        # item_li_1 = []
        added_items_1 = set()
        for item in item_list_1:
            if item in item_id_map1.keys():
                # item_li_1.append(item_id_map1[item])
                added_items_1.add(item_id_map1[item])
        item_li_1 = list(added_items_1)
        if len(item_li_1) >= num:
            with open(filepath + '/distinct_S.dat', 'a') as fout1:
                fout1.write(str(user_id))
                for t in item_li_1:
                    fout1.write(' '+str(t))
                fout1.write('\n')
            user_id_mapping_distinct1[u] = user_id  # 存储原userID和新的编号的对应关系
            user_id += 1
    print("distinct_S:{}".format(user_id), "!!! distinct_S is done !!!")
    user_id = common_users
    for u in tqdm(distinct_set_2, desc='distinct_T dat:', unit=' records'):
        item_list_2 = dict2[u]
        # item_li_2 = []
        added_items_2 = set()
        for item in item_list_2:
            if item in item_id_map2.keys():
                # item_li_2.append(item_id_map2[item])
                added_items_2.add(item_id_map2[item])
        item_li_2 = list(added_items_2)
        if len(item_li_2) >= num:
            with open(filepath + '/distinct_T.dat', 'a') as fout2:
                fout2.write(str(user_id))
                for t in item_li_2:
                    fout2.write(' '+str(t))
                fout2.write('\n')
            user_id_mapping_distinct2[u] = user_id  # 存储原userID和新的编号的对应关系
            user_id += 1
    print("distinct_T:{}".format(user_id), "!!! distinct_T is done !!!")

    userid_map1 = {}
    userid_map2 = {}

    for key, value in user_id_mapping_common.items():
        userid_map1[key] = value
    for key, value in user_id_mapping_distinct1.items():
        userid_map1[key] = value
    print("length of userid_map1", len(userid_map1))

    for key, value in user_id_mapping_common.items():
        userid_map2[key] = value
    for key, value in user_id_mapping_distinct2.items():
        userid_map2[key] = value
    print("length of userid_map2", len(userid_map2))

    return userid_map1, userid_map2, common_users

def fold_add(userid_map1, userid_map2, need_number, common_users_num, filepath):

    userid_map_S = {key: value for key, value in userid_map1.items() if int(value) < int(common_users_num + need_number)}
    userid_map_T = {key: value for key, value in userid_map2.items() if int(value) < int(common_users_num + need_number)}

    print("length of userid_map_S", len(userid_map_S))
    print("length of userid_map_T", len(userid_map_T))

    with open(filepath + '/distinct_S.dat', "r") as source_file:
        a = source_file.readlines()[:need_number]
    with open(filepath + '/common_S.dat', "r") as target_file:
        b = target_file.readlines()
    b.extend(a)
    with open(filepath + '/final_S.dat', 'w') as file_C:
        file_C.writelines(b)
    print("!!!common_S 记录已成功追加, final_S is done!!!")

    with open(filepath + '/distinct_T.dat', "r") as source_file:
        c = source_file.readlines()[:need_number]
    with open(filepath + '/common_T.dat', "r") as target_file:
        d = target_file.readlines()
    d.extend(c)
    with open(filepath + '/final_T.dat', 'w') as file_C:
        file_C.writelines(d)
    print("!!!common_T 记录已成功追加, final_T is done!!!")

    return userid_map_S, userid_map_T


def number_final_data(common_path, filepath, domain):
    # 读取原始数据文件
    with open(common_path) as file:
        lines = file.readlines()

    # 创建字典来映射原始itemID到新的itemID
    item_id_mapping = {}
    current_item_id = 0

    # 处理每一行数据
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        user_id = parts[0]
        new_items = []

        for item in parts[1:]:
            if item not in item_id_mapping:
                item_id_mapping[item] = str(current_item_id)
                current_item_id += 1
            new_items.append(item_id_mapping[item])

        new_line = [user_id] + new_items
        new_lines.append(" ".join(new_line))

    # 写入重新编号后的数据到新文件
    with open(filepath + '/result_{}.dat'.format(domain), "w") as new_file:
        new_file.write("\n".join(new_lines))
    return item_id_mapping

def change_gengeral_item_number(load_path, item_id_mapping, turnnum_path):
    # 1. 打开JSON文件并加载数据
    with open(load_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dict = []
    # 替换数据中的itemID
    for record in tqdm(data, desc='change_gengeral_item_number:', unit=' records'):
        if str(record["itemID"]) in item_id_mapping.keys():
            record["itemID"] = int(item_id_mapping[str(record["itemID"])])
        else:
            print("!!! no suit number !!!")

    # 保存替换后的数据为JSON文件
    with open(turnnum_path, "w") as f:
        json.dump(data, f, indent=4)
    return data

def necessary_dict(item_id_map, userid_map, pre_save_path, load_path):
    with open(pre_save_path) as f:
        data = json.load(f)
    necessary_dict = []
    style_count = 1
    style_type_map = {}
    for record in tqdm(data, desc='interact indexs:', unit=' records'):
        userID = record['reviewerID']
        itemID = record['asin']
        if userID in userid_map:
            user_id = userid_map[userID]
            if itemID in item_id_map:
                item_id = item_id_map[itemID]
                if 'vote' in record:
                    trust = record['vote']
                    if isinstance(trust, str):
                        trust = trust.replace(",", "")
                    else:
                        trust = trust
                else:
                    trust = "0"
                if record.get("style", {}).get("Format") or record.get("style", {}).get("Size"):
                    if "Format:" in record["style"]:
                        value = record["style"]["Format:"]
                    else:
                        value = record["style"]["Size:"]
                    if value not in style_type_map:
                        style_type_map[value] = style_count
                        category = style_count
                        style_count += 1
                    else:
                        category = style_type_map[value]
                else:
                    category = 0  # 只要category=0，就说明这条记录没有被划分类别

                container = {'userID': int(user_id), 'itemID': int(item_id), 'trust': int(trust),
                             'category': int(category), 'Timestamp': int(record['unixReviewTime'])}
                necessary_dict.append(container)

    with open(load_path, 'w') as f:
        json.dump(necessary_dict, f, indent=4)  # indent参数用于美化输出，可选
    return necessary_dict

def delete_overlapping_note(load_path, delete_path):
    with open(load_path) as f:
        data = json.load(f)
    # 用来记录已经出现的 ("userID", "itemID") 组合
    seen = set()
    # 用来存储去重后的数据
    unique_data = []
    for record in tqdm(data, desc='delete overlapping datas:', unit=' records'):
        user_item_pair = (record["userID"], record["itemID"])
        # 检查是否已经出现过该 ("userID", "itemID") 组合
        if user_item_pair not in seen:
            # 如果没有出现过，将记录添加到去重后的数据列表，并记录该组合
            unique_data.append(record)
            seen.add(user_item_pair)
    with open(delete_path, 'w') as f:
        json.dump(unique_data, f, indent=4)  # indent参数用于美化输出，可选
    return len(unique_data)
