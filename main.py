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

        return:
        无
    """
    if debug_log:
        print("进入函数：split_datasets()\n")
    # 删除已存在的 train_data.txt 和 validation_data.txt 文件
    if os.path.exists('train_data.txt'):
        os.remove('train_data.txt')
    if os.path.exists('validation_data.txt'):
        os.remove('validation_data.txt')

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
            with open('train_data.txt', 'a') as train_file:
                train_file.write(f"{user_id}|{len(train_items)}\n")
                for item in train_items:
                    train_file.write(f"{item[0]} {item[1]}\n")

            with open('validation_data.txt', 'a') as validation_file:
                validation_file.write(f"{user_id}|{len(validation_items)}\n")
                for item in validation_items:
                    validation_file.write(f"{item[0]} {item[1]}\n")

            line = file.readline().strip()
        # 在这里可以进行其他操作或返回结果
        # ...

    return


execute_split = True


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


execute_preprocess_user = True


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


execute_preprocess_item = True

mode = "test"  # 该mode下，需要将train划分，将train_data.txt作为训练集，validation_data.txt作为验证集
# mode = "report" # 该mode下，不需要将train划分，将train.txt作为训练集，test.txt作为测试集

if __name__ == "__main__":
    if mode == "test":
        # 数据预处理
        if execute_split:  # 如果不需要重新划分数据集，该变量请设置为false
            split_datasets("train.txt", 0.2, debug_log=True)
            preprocess_user("train_data.txt")
            preprocess_item("train_data.txt")

        # TODO()
    elif mode == "report":
        # 数据预处理
        if execute_preprocess_user:  # 如果已经生成了user_info.txt，就不用重复执行这个函数了，该变量请设置为false
            preprocess_user("train_data.txt")
        if execute_preprocess_item:  # 如果已经生成了item_info.txt，就不用重复执行这个函数了，该变量请设置为false
            preprocess_item("train_data.txt")

        # TODO()
