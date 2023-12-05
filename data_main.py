import utils.proprecssing as pp
from utils.parser import load_args

def handle(args):
    # 加载路径
    filepath = args.data_path + args.dataset + '/{}_{}'.format(args.domain_S, args.domain_T)

    pre_path_S = filepath + '/RawData/{}.json'.format(args.domain_S)
    pre_path_T = filepath + '/RawData/{}.json'.format(args.domain_T)
    pre_save_path_S = filepath + '/pre_{}.json'.format(args.domain_S)
    pre_save_path_T = filepath + '/pre_{}.json'.format(args.domain_T)  # 源数据处理完的数据

    load_path_S = filepath + '/general_{}.json'.format(args.domain_S)
    load_path_T = filepath + '/general_{}.json'.format(args.domain_T)
    delete_path_S = filepath + '/general_del_overlap_{}.json'.format(args.domain_S)
    delete_path_T = filepath + '/general_del_overlap_{}.json'.format(args.domain_T)

    common_path_S = filepath + '/final_S.dat'
    common_path_T = filepath + '/final_T.dat'

    turnnum_path_S = filepath + '/general_turnnum_{}.json'.format(args.domain_S)
    turnnum_path_T = filepath + '/general_turnnum_{}.json'.format(args.domain_T)

    # 源数据处理
    pp.pre_data(pre_path_S, pre_path_T, pre_save_path_S, pre_save_path_T, args.score, args.num)

    # 加载数据（为train和test打准备）
    user_item_index_S, user_item_index_T = pp.item_list(pre_save_path_S, pre_save_path_T)  # 得到项目一对多的字典
    item_id_map_S, item_id_map_T = pp.item_id_list(user_item_index_S, user_item_index_T)  # 创造项目索引的map，用于编号
    userid_map1, userid_map2, common_users_num = pp.overlap_data(item_id_map_S, item_id_map_T, user_item_index_S, user_item_index_T, args.num, filepath) # 创造用户索引的map，用于编号
    userid_map_S, userid_map_T = pp.fold_add(userid_map1, userid_map2, args.need_number, common_users_num, filepath)  # 得到最终用于训练的数据 以及 userID和索引的对应表

    # 为构建异构图做铺垫，如果要处理得到了general则注释代码
    necessary_dict_S = pp.necessary_dict(item_id_map_S, userid_map_S, pre_save_path_S, load_path_S)
    print("!!! general_{} is done !!!".format(args.domain_S), len(necessary_dict_S))
    necessary_dict_T = pp.necessary_dict(item_id_map_T, userid_map_T, pre_save_path_T, load_path_T)
    print("!!! general_{} is done !!!".format(args.domain_T), len(necessary_dict_T))

    # 将general文件中重复的记录只保留一条
    unique_len_S = pp.delete_overlapping_note(load_path_S, delete_path_S)
    unique_len_T = pp.delete_overlapping_note(load_path_T, delete_path_T)
    print("!!! general_del_overlap are done !!!", unique_len_S, unique_len_T)

    # 为了修改final.dat文件的item顺序
    item_id_mapping_S = pp.number_final_data(common_path_S, filepath, args.domain_S)
    item_id_mapping_T = pp.number_final_data(common_path_T, filepath, args.domain_T)
    # 修改顺序后general文件的对应的顺序同样需要改变
    general_turnnum_S = pp.change_gengeral_item_number(delete_path_S, item_id_mapping_S, turnnum_path_S)
    general_turnnum_T = pp.change_gengeral_item_number(delete_path_T, item_id_mapping_T, turnnum_path_T)
    print("!!! general_turnnum are done !!!", len(general_turnnum_S), len(general_turnnum_T))



if __name__ == '__main__':
    # hyper parameters
    args = load_args()
    print("—————————————— handle with datasets ——————————————")
    handle(args)


