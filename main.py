import os
from math import sqrt

import numpy as np
from scipy.sparse import coo_matrix
import sklearn
from scipy.stats import pearsonr
from random import random
from sklearn.metrics.pairwise import cosine_similarity

"""change
"""
max_item_id = 624960
u = 0


def split_datasets(file_name, validation_ratio, data_random=False, debug_log=False):
    """
        将数据集划分为训练集和验证集，并将训练集储存在train_data.txt中，
        验证集储存在validation_data.txt中。

        args:
            file_name: train.txt的文件路径
            validation_ratio (float): 验证集所占的比例，取值范围为 (0, 1)。
            data_random (bool): 是否打乱数据集顺序，默认为 False。
            debug_log(bool, 可选): 是否启用调试日志。默认为False。
        return:
        无
    """
    if debug_log:
        print("进入函数：split_datasets()\n")
    # 删除已存在的 train_data.txt 和 validation_data.txt 文件
    if os.path.exists('./temp/train_data.txt'):
        os.remove('./temp/train_data.txt')
    if os.path.exists('./temp/validation_data.txt'):
        os.remove('./temp/validation_data.txt')

    with open(file_name, 'r') as file:
        line = file.readline().strip()  # 读取第一行并去除首尾空格
        while line:
            user_id, num_items = line.split('|')
            temp = int(user_id)
            if debug_log:
                if temp % 3000 == 0:
                    print(f"split_datasets() User id: {temp}\n")
            num_items = int(num_items)
            # 读取 num_items 行数据并进行相关操作
            items = []
            # 读取 num_items 行数据并存储为列表
            for _ in range(num_items):
                item_id, score = file.readline().strip().split()
                items.append((item_id, score))

            # 打乱列表顺序
            if data_random:
                items.shuffle(items)

            # 将列表分割成训练集和验证集
            # split_index = int(num_items * validation_ratio)
            split_index = 1
            train_items = items[split_index:]
            validation_items = items[:split_index]

            # 将train_items和validation_items附加在文件后
            with open('./temp/train_data.txt', 'a') as train_file:
                train_file.write(f"{user_id}|{len(train_items)}\n")
                for item in train_items:
                    train_file.write(f"{item[0]}  {item[1]}\n")

            with open('./temp/validation_data.txt', 'a') as validation_file:
                validation_file.write(f"{user_id}|{len(validation_items)}\n")
                for item in validation_items:
                    validation_file.write(f"{item[0]}  {item[1]}\n")

            line = file.readline().strip()
    if debug_log:
        print("函数：split_datasets() 执行完成\n")

    return


def sparse_matrix_storage(file_name, output_file, debug_log=False):
    """
    将数据按照稀疏矩阵的形式存储。

    args:
        file_name(str): 存储数据的文件名。
        debug_log(bool, 可选): 是否启用调试日志。默认为False。

    return:
        无返回值
    """
    if debug_log:
        print("进入函数：sparse_matrix_storage()\n")
    with open(file_name, 'r') as file:
        line = file.readline().strip()  # 读取第一行并去除首尾空格
        while line:
            user_id, num_items = line.split('|')

            if debug_log:
                temp = int(user_id)
                if temp % 3000 == 0:
                    print(f"sparse_matrix_storage() User id: {temp}\n")

            num_items = int(num_items)
            # 读取 num_items 行数据并进行相关操作
            for _ in range(num_items):
                item_id, score = file.readline().strip().split()
                output_file.write(f"{user_id} {item_id} {score}\n")
            line = file.readline().strip()
    if debug_log:
        print("函数：sparse_matrix_storage() 执行完成\n")
    return


correlation_matrix = []


def compute_user_similarity(file_name, sim_func="pearson", debug_log=False):
    """
        计算user-user之间的相似度，将结果存在文件user_similarity.txt中
        （可以存在内存，但是存在文件中的话方便多个人协作写代码，在内存里的话因为互相之间不知道数据格式，可能改起来比较麻烦）
          1 2 3 4 5
        1 1 w x y z
        2 a 1 b c d
        3 ...
        4
        5
        文件中的格式为：
        1 w x y z
        a 1 b c d
        ...
        (两个元素之间是空格，矩阵中的一行对应文件中的一行 一行中有10000多个数字，应该可以的吧)

        args:
            file_name: sparse_matrix.txt的文件路径
            sim_func: 选取的相似计算函数
            debug_log(bool, 可选): 是否启用调试日志。默认为False。
        return:
            correlation_matrix(class:numpy.ndarray, shape:(19835, 19835))：用户之间的相关系数矩阵（对角线元素为1，对称矩阵）
    """
    if debug_log:
        print("进入函数：compute_user_similarity()\n")

    # 读取用户均值文件
    user_avg_scores = {}

    with open('./temp/user_info.txt', 'r') as file:
        for line in file:
            user_id, avg_score = line.strip().split(':')
            user_id = int(user_id.strip())
            user_avg_scores[user_id] = float(avg_score)

    if debug_log:
        print("compute_user_similarity() 读取用户均值结束\n")

    # 读取用户-物品-评价表文件并计算相关系数
    num_users = 19835
    user_ids = []
    item_ids = []
    scores = []  # score - avg_userid

    with open(file_name, 'r') as file:
        for line in file:
            user_id, item_id, score = line.strip().split()
            user_id = int(user_id)
            item_id = int(item_id)
            score = float(score)

            user_ids.append(user_id)
            item_ids.append(item_id)
            # scores.append(score - user_avg_scores[user_id])
            scores.append(score)

    if debug_log:
        print("compute_user_similarity() 读取三元组结束\n")

    # 构建评价矩阵
    rating_matrix = coo_matrix((scores, (user_ids, item_ids)), shape=(num_users, max(item_ids) + 1))

    # 计算相关系数矩阵 跑完大概要3000天,你是认真的吗
    """
    correlation_matrix = np.zeros((num_users, num_users))

    for i in range(num_users):
        for j in range(i, num_users):
            if debug_log:
                print(f"compute_user_similarity() user_id1 : {i} user_id2 : {j}\n")
            user1_ratings = rating_matrix.getrow(i).toarray()[0]
            user2_ratings = rating_matrix.getrow(j).toarray()[0]
            valid_ratings = np.logical_and(user1_ratings != 0, user2_ratings != 0)

            if np.sum(valid_ratings) > 1:
                correlation, _ = pearsonr(user1_ratings[valid_ratings], user2_ratings[valid_ratings])
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
    """
    global correlation_matrix
    # cosine相关系数矩阵生成：用时1min以内
    correlation_matrix = cosine_similarity(rating_matrix)

    if debug_log:
        print("函数: compute_user_similarity() correlation_matrix生成完毕\n")

    # 存储相关系数矩阵到文件 用时10分钟以上，文件大小9G（一个浮点数写入文件后占很多字符，太大了实在）
    # if os.path.exists('./temp/user_similarity.txt'):
    #    os.remove('./temp/user_similarity.txt')
    # np.savetxt('./temp/user_similarity.txt', correlation_matrix, delimiter=' ')

    if debug_log:
        print("函数: compute_user_similarity() 执行完毕\n")
    print(correlation_matrix.shape)
    return correlation_matrix


def predict(file_name, k=10, debug_log=False, predict_mode=0):
    """
        读取要测试的文件，生成测试结果。
        注意，要测试的文件可能是test.txt或者validation_data.txt，格式分别如下：
            validation_data.txt：
                <user id>|<numbers of rating items>
                <item id>   <score>
            test.txt
                <user id>|<numbers of rating items>
                <item id>
            （为了验证方便，在validation_data.txt保留了score，需要在读取文件时忽略   <score>，具体
            怎么忽略自行考虑）

            生成的结果格式 result.txt：
            <user id>|<numbers of rating items>
            <item id>   <score>
        args:
            file_name: 要测试的txt的文件路径
            k: k个醉相思的用户
        return:
            无
    """

    if debug_log:
        print("进入函数：predict()\n")
    # TODO() 将user_similarity.txt读取到内存（也许可以逐行读取，每次对该行的相似值进行排序，以便处理k个没有打分的情况）

    # TODO() 获取全部评分均值u，各用户的评分均值，各物品的评分均值

    # TODO() 打开file_name文件，获取要测试的数据

    # TODO() 逐个计算并写入文件，如果不想重复打开关闭输入输出文件流，可修改函数参数
    # pred(u, i) = u + b_u + b_i + 前k个的加权平均（如果前k个没有打分的话要怎么计算捏？）

    if debug_log:
        print("predict() 开始读取用户数据\n")
    # score_sum = 0.0
    # score_count = 0
    users_id = []
    user_score = []
    with open('./temp/user_info.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            cur_user_id, cur_user_score = line.split(' : ')
            cur_user_id = int(cur_user_id)
            cur_user_score = float(cur_user_score)
            # score_sum += cur_user_score
            # score_count += 1
            users_id.append(cur_user_id)
            user_score.append(cur_user_score)
    users_id = np.array(users_id)
    user_score = np.array(user_score)
    # u = score_sum / score_count
    if debug_log:
        print("predict() 读取用户数据完毕\n")

    print('users_id shape:', users_id.shape)
    print('user_score shape:', user_score.shape)
    # print('u:', u)

    if debug_log:
        print("predict() 开始读取物品数据\n")
    items_id = []
    item_score = []
    with open('./temp/item_info.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            cur_item_id, cur_item_score = line.split(' : ')
            cur_item_id = int(cur_item_id)
            cur_item_score = float(cur_item_score)
            items_id.append(cur_item_id)
            item_score.append(cur_item_score)
    items_id = np.array(items_id)
    item_score = np.array(item_score)
    if debug_log:
        print("predict() 读取物品数据完毕\n")

    print('items_id shape:', items_id.shape)
    print('item_score shape:', item_score.shape)

    if debug_log:
        print("predict() 开始读取历史打分文件\n")
    history_user_id = []
    history_item_id = []
    history_score = []
    with open('./temp/sparse_matrix.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            cur_user_id, cur_item_id, cur_score = line.split(' ')
            cur_user_id = int(cur_user_id)
            cur_item_id = int(cur_item_id)
            cur_score = int(cur_score)
            history_user_id.append(cur_user_id)
            history_item_id.append(cur_item_id)
            history_score.append(cur_score)
    history_user_id = np.array(history_user_id)
    history_item_id = np.array(history_item_id)
    history_score = np.array(history_score)
    if debug_log:
        print("predict() 读取历史打分文件完毕\n")
    print('h_user shape:', history_user_id.shape)
    print('h_item shape:', history_item_id.shape)
    print('h_score shape:', history_score.shape)

    if debug_log:
        print("predict() 开始读取测试文件\n")
    test_users_id = []
    test_items_id = []
    test_cases = [0] * 19835
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip('\n')
            # cur_user_id = -1
            if '|' in line:  # first line: <user id>|<numbers of rating items>
                cur_user_id, cur_rating_count = line.split('|')
                cur_user_id = int(cur_user_id)
                cur_rating_count = int(cur_rating_count)
                # test_users_id.append(cur_user_id)
                for i in range(cur_rating_count):
                    test_users_id.append(cur_user_id)
                test_cases[cur_user_id] = cur_rating_count
            else:
                if predict_mode == 0:  # second line: in validation_data:<item id>   <score>
                    cur_item_id, _ = line.split('  ')
                    cur_item_id = int(cur_item_id)
                    test_items_id.append(cur_item_id)
                    # cur_user_id = test_users_id[-1]
                    # test_users_id.append(cur_user_id)
                elif predict_mode == 1:  # second line: in test:<item id>
                    cur_item_id = line
                    cur_item_id = int(cur_item_id)
                    test_items_id.append(cur_item_id)
                    # cur_user_id = test_users_id[-1]
                    # test_users_id.append(cur_user_id)

    test_users_id = np.array(test_users_id)
    test_items_id = np.array(test_items_id)
    # test_user_item = np.array([test_users_id, test_items_id])
    # test_user_item = np.transpose(test_user_item)
    if debug_log:
        print("predict() 读取测试文件完毕\n")
    # print('test_user_item shape:', test_user_item.shape)
    print('test_users_id shape:', test_users_id.shape)
    print('test_items_id shape:', test_items_id.shape)

    if debug_log:
        print("predict() 开始计算得分\n")
    global correlation_matrix, u
    predict_score = []
    for i in range(test_items_id.shape[0]):
        if debug_log:
            print(f"predict() 计算用户{i}\n")
        cur_user_id = test_users_id[i]
        cur_item_id = test_items_id[i]
        cur_user_similarity = correlation_matrix[cur_user_id, :]  # 取出当前用户的所有用户相似度
        cur_user_similarity = np.array(cur_user_similarity)

        # print('cur_user_similarity ', cur_user_similarity)

        sorted_similarity = sorted(enumerate(cur_user_similarity), key=lambda x: x[1], reverse=True)
        similarity_user_id = [j[0] for j in sorted_similarity]  # 排序并返回相应原下标

        k_similarity = sorted_similarity[1: k+1]  # 删去1
        # print(k_similarity[0])
        numerator_sum = 0  # 分子和：(r_v_i - r_v_avg)*s_u_v
        denominator_sum = 0  # 分母和：s_u_v
        for k_i in range(k):
            cur_similarity = k_similarity[k_i][1]
            # cur_similarity = float(cur_similarity)
            # print('cur_similarity:', cur_similarity)
            denominator_sum += cur_similarity
            user_v_id = similarity_user_id[k_i]
            user_v_avg_score = user_score[user_v_id]  # r_v: 用户v打分的平均分
            # user_v_socre_i = history_score
            user_v_score = 0  # r_v_i: 用户v对物品i的打分历史
            # print('user_v_id', user_v_id)
            # print(np.where(history_user_id == user_v_id))
            user_v_list = np.where(history_user_id == user_v_id)
            for v_i in user_v_list:
                # print('v_i', v_i)
                # print('history_item_id', history_item_id[v_i][0])
                if history_item_id[v_i][0] == cur_item_id:
                    user_v_score += history_score[v_i][0]  # +=是因为循环内变量若赋值会被识别为新变量 有点丑陋但暂时没想到怎么改
                    # print('user_v_score before', user_v_score)
                    break
            # 用户v没有对物品i打过分，选择该用户的历史平均分
            user_v_score = user_v_avg_score if user_v_score == 0 else user_v_score
            # print('user_v_score after', user_v_score)
            numerator_sum += (user_v_score - user_v_avg_score) * cur_similarity
            # print(type(user_v_score), type(user_v_avg_score), type(cur_similarity))
        bias_u = user_score[cur_user_id] - u
        # print(bias_u)
        # print('cur_item_id', cur_item_id)
        # print('items_id', items_id)
        # print(np.where(items_id == cur_item_id)[0])
        # print('type:', type((np.where(items_id == cur_item_id))[0]))
        bias_i = 0.0
        if cur_item_id in items_id:
            tmp_index = np.where(items_id == cur_item_id)[0][0]
            print('tmp_index:', tmp_index)
            bias_i += item_score[tmp_index] - u
        bias_i = float(bias_i)
        # print('bias_i type', type(bias_i))
        print('bias_i', bias_i)
        res_score = numerator_sum / denominator_sum + u + bias_u + bias_i
        # print(type(numerator_sum), type(denominator_sum), type(u), type(bias_u), type(bias_i))

        # print('res_score type', type(res_score))
        res_score = float(res_score)
        predict_score.append(res_score)
    predict_score = np.array(predict_score)
    if debug_log:
        print("predict() 计算得分完毕\n")

    print('predict_score shape:', predict_score.shape)

    if debug_log:
        print("predict() 开始写result文件\n")

    if os.path.exists('./temp/result.txt'):
        os.remove('./temp/result.txt')

    """
    with open('./temp/result.txt', 'a') as f:
        index = 0
        while index < test_items_id.shape[0]:
            rating_nums = len(np.argwhere(test_users_id == test_users_id[index])[:0])
            first_line = str(int(test_users_id[index])) + '|' + str(int(rating_nums))
            f.write(first_line)
            for i in range(rating_nums):
                tmp_str = str(int(test_items_id[index+i])) + ' ' + str(int(predict_score[index+i]))
                f.write(tmp_str)
            index += rating_nums
    """
    with open('./temp/result.txt', 'a') as f:
        index = 0
        for user in range(19835):
            rating_nums = test_cases[user]
            first_line = str(int(user)) + '|' + str(rating_nums) + '\n'
            f.write(first_line)
            for i in range(rating_nums):
                tmp_str = str(test_items_id[index + i]) + ' ' + str(predict_score[index + i]) + '\n'
                f.write(tmp_str)
            index += rating_nums

    if debug_log:
        print("predict() 写result文件完毕\n")

    if debug_log:
        print("函数：predict() 执行完毕\n")
    return


def evaluate(val, res):
    """
        比较validate_data.txt的结果与result.txt中的结果，计算RMSE
    args:
        None
    return:
        无
    """
    RMSE = float(0)
    count = 0
    line_val = val.readline().strip()  # 读取第一行并去除首尾空格
    line_res = res.readline().strip()  # 读取第一行并去除首尾空格
    while line_val:
        user_id, num_items = line_val.split('|')
        user_id_r, num_items_r = line_res.split('|')
        user_id = int(user_id)
        user_id_r = int(user_id_r)
        num_items = int(num_items)
        num_items_r = int(num_items)
        assert (user_id == user_id_r)
        assert (num_items == num_items_r)
        count += num_items
        for _ in range(num_items):
            item_id, score = val.readline().strip().split()
            item_id_r, score_r = res.readline().strip().split()
            assert (item_id == item_id_r)
            print(f"item_id: {score} item_id_r:{score_r}")
            RMSE += (float(score) - float(score_r)) * (float(score) - float(score_r))
            print(f"RMSE: {RMSE}")
        line_val = val.readline().strip()
        line_res = res.readline().strip()
    print(f"item_id: {count}")
    RMSE = sqrt(RMSE / count)
    print(f"最终测试结果: test数量：{count}, RMSE = {RMSE}")
    return RMSE


def preprocess_user(file_name, debug_log=False):
    """
        预处理user，统计user信息，并存储为id : avg_score的形式，输出文件名字为user_info.txt

        args:
            file_name: train.txt或者train_data.txt的文件路径
        return:
            无
    """
    # 3.统计user（方便的话，统计各user的打分平均值，格式为id : 均值）

    if debug_log:
        print("进入函数：preprocess_user()\n")

    if os.path.exists('./temp/user_info.txt'):
        os.remove('./temp/user_info.txt')
    # 可能可以合并于划分train_data.txt时，
    # 具体做法即声明一个变量，在得到train_data.txt中新的rate_num时把这个rate_num读取存储下来，
    # 最后形式是一个一维数组，每一元素代表某一用户打分item的个数
    rate_nums = []
    print("preprocess_user() 开始记录用户打分\n")
    with open(file_name, 'r') as f:  # 获取每个用户对几个item进行了评分
        for line in f:
            line = line.strip('\n')
            if '|' in line:
                _, cur_user_rate_nums = line.split('|')
                cur_user_rate_nums = int(cur_user_rate_nums)
                rate_nums.append(cur_user_rate_nums)
    rate_nums = np.array(rate_nums)
    # print("rate_nums.shape: ", rate_nums.shape)

    # 处理已经变成三元组的数据文件，得到每个用户的打分均值
    users_id = []
    score_sum = []
    final_user_id = 0
    final_score_sum = 0
    tot_score = 0
    tot_count = 0

    with open('./temp/sparse_matrix.txt', 'r') as f:  # 已处理成三元组
        last_user_id = 0
        cur_score_sum = 0
        for line in f:
            line = line.strip('\n')
            cur_user_id, _, cur_item_score = line.split()  # 空格隔开
            cur_user_id = int(cur_user_id)
            cur_item_score = int(cur_item_score)

            # 用于统计全局打分
            tot_score += cur_item_score
            tot_count += 1

            if last_user_id != cur_user_id:  # 更换用户，可以汇总刚才的统计数据
                last_user_id = int(last_user_id)
                cur_score_sum = int(cur_score_sum)
                users_id.append(last_user_id)
                score_sum.append(cur_score_sum)
                '''
                if last_user_id != -1:
                    users_id.append(last_user_id)
                    score_sum.append(cur_score_sum)
                '''
                last_user_id = cur_user_id
                cur_score_sum = cur_item_score
            else:
                cur_score_sum += cur_item_score  # 统计每个用户打分的总数
            final_user_id = cur_user_id
            final_score_sum = cur_score_sum
    # 处理最后一个用户
    users_id.append(final_user_id)
    score_sum.append(final_score_sum)
    # print("last ", final_user_id, final_score_sum)

    users_id = np.array(users_id)
    score_sum = np.array(score_sum)
    # print("user_id.shape: ", users_id.shape)
    # print("score_sum.shape: ", score_sum.shape)
    users_id = users_id.astype(np.float64)
    score_sum = score_sum.astype(np.float64)
    avg_score = score_sum / rate_nums
    res_matrix = np.array([users_id, avg_score])
    res_matrix = np.transpose(res_matrix)  # e.g: [[u1, avg_score1], [u2, avg_score2]]

    # 统计全局打分均分
    global u
    u = tot_score / tot_count

    print("preprocess_user() 记录用户打分完成\n")

    # 写入user_info
    print("preprocess_user() 开始写入文件\n")
    for i in range(res_matrix.shape[0]):
        tmp_str = str(int(res_matrix[i][0])) + ' : ' + str(res_matrix[i][1]) + '\n'
        with open('./temp/user_info.txt', 'a') as file:
            file.write(tmp_str)
    print("preprocess_user() 写入文件完成\n")
    if debug_log:
        print("函数：preprocess_user() 执行完成\n")
    return


def preprocess_item(file_name, debug_log=False):
    """
        预处理item，统计item信息，并存储为id : avg_score的形式，输出文件名字为item_info.txt

        args:
            file_name: train.txt或者train_data.txt的文件路径（如果用item_attr的话，改一下注释）
        return:
            无
    """

    if debug_log:
        print("进入函数：preprocess_item()\n")
    if os.path.exists('./temp/item_info.txt'):
        os.remove('./temp/item_info.txt')
    items_id = np.zeros(max_item_id + 1, dtype=int)
    item_score_count = np.zeros(max_item_id + 1, dtype=int)
    item_score_sum = np.zeros(max_item_id + 1, dtype=int)

    # 记录得分
    print("preprocess_item() 开始记录物品得分\n")
    with open(file_name, 'r') as f:
        for line in f:
            if '|' in line:
                pass
            else:
                cur_item_id, cur_item_score = line.split()
                cur_item_id = int(cur_item_id)
                cur_item_score = int(cur_item_score)
                items_id[cur_item_id] = cur_item_id
                item_score_sum[cur_item_id] += cur_item_score
                item_score_count[cur_item_id] += 1
    items_id = np.array(items_id)
    item_score_sum = np.array(item_score_sum)
    item_score_count = np.array(item_score_count)
    exist_item_id = []
    exist_score_sum = []
    exist_score_count = []
    for i in range(max_item_id + 1):
        if item_score_count[i] != 0:
            exist_item_id.append(items_id[i])
            exist_score_count.append(item_score_count[i])
            exist_score_sum.append(item_score_sum[i])
    exist_score_count = np.array(exist_score_count)
    exist_score_sum = np.array(exist_score_sum)
    exist_item_id = np.array(exist_item_id)
    avg_score = exist_score_sum / exist_score_count
    # print("item_score_sum shape", item_score_sum.shape)
    # print("item_score_count shape", item_score_count.shape)
    res_matrix = np.array([exist_item_id, avg_score])
    res_matrix = np.transpose(res_matrix)
    print("preprocess_item() 记录物品得分完成\n")

    # 写入item_info
    print("preprocess_item() 开始写入文件\n")
    for i in range(res_matrix.shape[0]):
        tmp_str = str(int(res_matrix[i][0])) + ' : ' + str(res_matrix[i][1]) + '\n'
        with open('./temp/item_info.txt', 'a') as file:
            file.write(tmp_str)
    print("preprocess_item() 写入文件完成\n")
    if debug_log:
        print("函数：preprocess_item() 执行完成\n")
    return


#  预处理部分
execute_split = False
execute_matrix_storage = False
execute_preprocess_user = True # 涉及global u的计算，必须运行这个
execute_preprocess_item = False
#  训练部分
execute_compute_similarity = True
execute_predict = True
execute_evaluate = True

mode = "test"  # 该mode下，需要将train划分，将train_data.txt作为训练集，validation_data.txt作为验证集
# mode = "report" # 该mode下，不需要将train划分，将train.txt作为训练集，test.txt作为测试集，没有RMSE计算了哦

if __name__ == "__main__":
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    if mode == "test":
        # 数据预处理
        if execute_split:  # 如果不需要重新划分数据集，该变量请设置为false
            split_datasets("train.txt", 0.01, debug_log=True)

        if execute_matrix_storage:  # 如果已经生成了sparse_matrix.txt，就不用重复执行这个函数了
            if os.path.exists('./temp/sparse_matrix.txt'):
                os.remove('./temp/sparse_matrix.txt')
            with open("./temp/sparse_matrix.txt", "a") as res_file:
                sparse_matrix_storage("./temp/train_data.txt", res_file, debug_log=True)

        if execute_preprocess_user:  # 如果已经生成了item_info.txt，就不用重复执行这个函数了，该变量请设置为false
            preprocess_user("./temp/train_data.txt", debug_log=True)

        if execute_preprocess_item:  # 如果已经生成了item_info.txt，就不用重复执行这个函数了，该变量请设置为false
            preprocess_item("./temp/train_data.txt", debug_log=True)

        # 训练
        if execute_compute_similarity:
            compute_user_similarity("./temp/sparse_matrix.txt", debug_log=True)
        if execute_predict:
            predict("./temp/validation_data.txt", debug_log=True, predict_mode=0)

        # 测试
        if execute_evaluate:
            with open("./temp/validation_data.txt", "r") as validate_file:
                with open("./temp/result.txt", "r") as result_file:
                    evaluate(validate_file, result_file)

    elif mode == "report":
        # 数据预处理
        if os.path.exists('./temp/sparse_matrix.txt'):
            os.remove('./temp/sparse_matrix.txt')
        if execute_matrix_storage:  # 如果已经生成了sparse_matrix.txt，就不用重复执行这个函数了
            with open("./temp/sparse_matrix.txt", "a") as output_file:
                sparse_matrix_storage("train.txt", output_file, debug_log=True)
        if execute_preprocess_user:  # 如果已经生成了user_info.txt，就不用重复执行这个函数了，该变量请设置为false
            preprocess_user("train.txt")
        if execute_preprocess_item:  # 如果已经生成了item_info.txt，就不用重复执行这个函数了，该变量请设置为false
            preprocess_item("train.txt")

        # 训练
        if execute_compute_similarity:
            compute_user_similarity("./temp/sparse_matrix.txt")
        if execute_predict:
            predict("./temp/test.txt", predict_mode=1)
