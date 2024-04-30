import pandas as pd
import os
import shutil
import csv
import random
import argparse


def mkdir_if_not_exist(dir_name, is_delete=False):
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return True
    except Exception as e:
        return False
    

def sort_two_list(list1, list2, reverse=False):
    """
    排序两个列表
    :param list1: 列表1
    :param list2: 列表2
    :param reverse: 逆序
    :return: 排序后的两个列表
    """
    try:
        list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2), reverse=reverse)))
    except Exception as e:
        sorted_id = sorted(range(len(list1)), key=lambda k: list1[k], reverse=True)
        list1 = [list1[i] for i in sorted_id]
        list2 = [list2[i] for i in sorted_id]

    return list1, list2


def traverse_dir_files(root_dir, suffix=None, is_sorted=True):
    """
    列出文件夹中的文件, 深度遍历
    :param root_dir: 根目录
    :param suffix: 后缀名
    :param is_sorted: 是否排序，耗时较长
    :return: [文件路径列表, 文件名称列表]
    """
    names_list = []
    paths_list = []
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith('.'):  # 去除隐藏文件
                continue
            if suffix:  # 根据后缀名搜索
                if name.endswith(tuple(suffix)):
                    names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                names_list.append(name)
                paths_list.append(os.path.join(parent, name))
    if not names_list:  # 文件夹为空
        return paths_list, names_list
    if is_sorted:
        paths_list, names_list = sort_two_list(paths_list, names_list)
    return paths_list, names_list


def make_labels_isic(dataset='ISIC2018',label_path=None,ori_path=None,target_path=None):
    label_df = pd.read_csv(label_path)

    label_dicts = {
        "ISIC2018": {'MEL': 'MEL', 'NV': 'NV', 'BCC': 'BCC', 'AKIEC': 'AKIEC', 'BKL': 'BKL', 'DF': 'DF', 'VASC': 'VASC'}}
    label_dict = label_dicts[dataset]
    for idx, row in label_df.iterrows():
        name = row['image']
        for label in label_dict.keys():
            if row[label] == 1:
                new_path = os.path.join(target_path, label_dict[label])
                mkdir_if_not_exist(new_path)
                new_img_path = os.path.join(new_path, f"{name}.jpg")
                shutil.copy(os.path.join(ori_path, f"{name}.jpg"), new_img_path)

    print("finish!")

def split_dataset(path,dataset='ISIC2018'):
    # image,category,label 3:1:1
    data_path = os.path.join(path,'{}_Dataset'.format(dataset))
    category = os.listdir(data_path) 
    #train.csv,val.csv,test.csv
    f_train = open(os.path.join(path,'{}_train.csv'.format(dataset)), 'a+', newline='')
    f_val = open(os.path.join(path,'{}_val.csv'.format(dataset)), 'a+', newline='')
    f_test = open(os.path.join(path,'{}_test.csv'.format(dataset)), 'a+', newline='')

    train_writer = csv.writer(f_train)
    val_writer = csv.writer(f_val)
    test_writer = csv.writer(f_test)

    headers = ['image','category','label']
    train_writer.writerow(headers)
    val_writer.writerow(headers)
    test_writer.writerow(headers)

    train_writer = csv.writer(f_train)
    val_writer = csv.writer(f_val)
    test_writer = csv.writer(f_test)

    for label,cate in enumerate(category):
        dir_path = os.path.join(data_path,cate)
        paths_list,_ = traverse_dir_files(dir_path)

        random.shuffle(paths_list)
        for idx,path in enumerate(paths_list):
            row = []
            img_name = path.split('/')[-1].split('.')[0]
            row.append(img_name)
            row.append(cate)
            row.append(label)
            if idx % 5 ==0:
                test_writer.writerow(row)
            elif idx % 5 ==1:
                val_writer.writerow(row)
            else:
                train_writer.writerow(row)
    f_train.close()
    f_val.close()
    f_test.close()
    print("finish!")

parser = argparse.ArgumentParser(description='preprocess the dataset donwloaded from ISIC')
parser.add_argument('--dataset',default='ISIC2018',type=str,help='ISIC2018')
parser.add_argument('--datapath',default='./data/ISIC2018',type=str,help='the path of dataset')

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    datapath = args.datapath
    label_path = os.path.join(datapath,'ISIC2018_Task3_Training_GroundTruth.csv')
    ori_path = os.path.join(datapath, 'ISIC2018_Task3_Training_Input')
    target_path = os.path.join(datapath, 'ISIC2018_Dataset')
    make_labels_isic(dataset=dataset,label_path=label_path,ori_path=ori_path,target_path=target_path)
    split_dataset(path=datapath,dataset=dataset)
