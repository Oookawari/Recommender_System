def split_datasets(file_name, validation_ratio):
    """
        将数据集划分为训练集和验证集，并将训练集储存在train_data.txt中，
        验证集储存在validation_data.txt中。

        args:
            file_name: train.txt的文件路径
            validation_ratio (float): 验证集所占的比例，取值范围为 (0, 1)。

        return:
        无
    """
    # TODO()
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
            split_datasets("train.txt", 0.2)
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
