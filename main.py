import os
from random import random


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
            split_index = int(num_items * validation_ratio)
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


def compute_user_similarity(file_name, sim_func="pearson"):
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
        return:
            无
    """

    # TODO()
    def pearson_correlation(xxx, yyy):
        # TODO()
        return

    def cosine_correlation(xxx, yyy):
        # TODO()
        return

    return


def predict(file_name, k=10):
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
    # TODO() 将user_similarity.txt读取到内存（也许可以逐行读取，每次对该行的相似值进行排序，以便处理k个没有打分的情况）

    # TODO() 获取全部评分均值u，各用户的评分均值，各物品的评分均值

    # TODO() 打开file_name文件，获取要测试的数据

    # TODO() 逐个计算并写入文件，如果不想重复打开关闭输入输出文件流，可修改函数参数
    # pred(u, i) = u + b_u + b_i + 前k个的加权平均（如果前k个没有打分的话要怎么计算捏？）
    return


def evaluate():
    """
        比较validate_data.txt的结果与result.txt中的结果，计算RMSE之类的
    args:
        None
    return:
        无
    """
    return


def preprocess_user(file_name):
    """
        预处理user，统计user信息，并存储为id : avg_score的形式，输出文件名字为user_info.txt

        args:
            file_name: train.txt或者train_data.txt的文件路径
        return:
            无
    """
    # TODO()
    return


def preprocess_item(file_name):
    """
        预处理item，统计item信息，并存储为id : avg_score的形式，输出文件名字为item_info.txt

        args:
            file_name: train.txt或者train_data.txt的文件路径（如果用item_attr的话，改一下注释）
        return:
            无
    """
    # TODO()
    return


#  预处理部分
execute_split = True
execute_matrix_storage = True
execute_preprocess_user = True
execute_preprocess_item = True
execute_compute_similarity = True
execute_predict = True

mode = "test"  # 该mode下，需要将train划分，将train_data.txt作为训练集，validation_data.txt作为验证集
# mode = "report" # 该mode下，不需要将train划分，将train.txt作为训练集，test.txt作为测试集

if __name__ == "__main__":
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    if mode == "test":
        # 数据预处理
        if execute_split:  # 如果不需要重新划分数据集，该变量请设置为false
            split_datasets("train.txt", 0.2, debug_log=True)

        if execute_matrix_storage:  # 如果已经生成了sparse_matrix.txt，就不用重复执行这个函数了
            if os.path.exists('./temp/sparse_matrix.txt'):
                os.remove('./temp/sparse_matrix.txt')
            with open("./temp/sparse_matrix.txt", "a") as res_file:
                sparse_matrix_storage("./temp/train_data.txt", res_file, debug_log=True)

        if execute_preprocess_item:  # 如果已经生成了item_info.txt，就不用重复执行这个函数了，该变量请设置为false
            preprocess_item("train_data.txt")

        if execute_preprocess_item:  # 如果已经生成了item_info.txt，就不用重复执行这个函数了，该变量请设置为false
            preprocess_item("train_data.txt")

        # 训练
        if execute_compute_similarity:
            compute_user_similarity("./temp/sparse_matrix.txt")
        if execute_predict:
            predict("./temp/validation_data.txt")
        evaluate()

    elif mode == "report":
        # 数据预处理
        if os.path.exists('sparse_matrix.txt'):
            os.remove('sparse_matrix.txt')
        if execute_matrix_storage:  # 如果已经生成了sparse_matrix.txt，就不用重复执行这个函数了
            with open("sparse_matrix.txt", "a") as output_file:
                sparse_matrix_storage("train.txt", output_file, debug_log=True)
        if execute_preprocess_user:  # 如果已经生成了user_info.txt，就不用重复执行这个函数了，该变量请设置为false
            preprocess_user("train.txt")
        if execute_preprocess_item:  # 如果已经生成了item_info.txt，就不用重复执行这个函数了，该变量请设置为false
            preprocess_item("train.txt")

        # 训练
        if execute_compute_similarity:
            compute_user_similarity("./temp/sparse_matrix.txt")
        if execute_predict:
            predict("./temp/test.txt")
