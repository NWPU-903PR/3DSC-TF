import os
import sys
import json
import pickle
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from patchify import patchify
from scipy.stats import ttest_ind


def read_split_data_tvt_5cv(image_train, image_test, root: str, val_rate: float = 0.2):
    # fold是五折交叉验证中的折数
    # 分出验证集和训练集
    random.seed(10)  # 保证随机结果可复现 随机数相同的情况下划分的训练集和测试集是一样的
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_file = open(image_train,"r")
    test_file = open(image_test,"r")
    train_list = train_file.readlines()
    test_list = test_file.readlines()
    every_class_num = len(train_list)+len(test_list)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    test_images_path = []  # 存储测试集的所有图片路径
    test_images_label = [] # 存储测试集图片对应索引信息

    for i in test_list:
        image_path = os.path.join(root,i[:-1])
        if str(i)[0:2] == "AD":

            image_label = 1
        else:
            image_label = 0
        test_images_path.append(image_path)
        test_images_label.append(image_label)


    image_val = random.sample(train_list,k=int(len(train_list)*val_rate))

    for i in train_list:
        image_path = os.path.join(root,i[:-1])
        if str(i)[0:2] == "AD":

            image_label = 1
        else:
            image_label = 0

        if i in image_val:  # 如果该路径在采样的验证集样本中则存入验证集
            val_images_path.append(image_path)
            val_images_label.append(image_label)
        else:  # 否则存入训练集
            train_images_path.append(image_path)
            train_images_label.append(image_label)


    print("{} images were found in the dataset.".format(every_class_num))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images for test.".format(len(test_images_path)))

    return train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label
# data_path = "/home/disk1/zhouqinyi/SMALLDATASET/ADNC"
# train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data_tvt_5cv(1,data_path)


def read_split_data_tvt_5cv_test(image_test, root: str, val_rate: float = 0.2):
    # image_test是测试集的csv文件
    # 分出验证集和训练集
    random.seed(0)  # 保证随机结果可复现 随机数相同的情况下划分的训练集和测试集是一样的
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    test_file = open(image_test,"r")

    test_list = test_file.readlines()

    test_images_path = []  # 存储测试集的所有图片路径
    test_images_label = [] # 存储测试集图片对应索引信息

    for i in test_list:
        image_path = os.path.join(root,i[:-1])
        if str(i)[0:2] == "AD":
            image_label = 1
        else:
            image_label = 0
        test_images_path.append(image_path)
        test_images_label.append(image_label)
    print("{} images for test.".format(len(test_images_path)))

    return test_images_path, test_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    train_preds = []
    train_trues = []
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        train_preds.extend(pred_classes.detach().cpu().numpy())
        train_trues.extend(labels.detach().cpu().numpy())

    confusion = confusion_matrix(train_trues,train_preds)
    TP = confusion[1,1]
    TN = confusion[0,0]
    FP = confusion[0,1]
    FN = confusion[1,0]

    sklearn_specificity = TN/float(TN+FP)
    sklearn_accuracy = accuracy_score(train_trues, train_preds)
    sklearn_precision = precision_score(train_trues, train_preds, average='weighted')
    sklearn_recall = sklearn_recall = TP / float(TP + FN)
    sklearn_f1 = f1_score(train_trues, train_preds, average='weighted')
    sklearn_auc = metrics.roc_auc_score(train_trues, train_preds, multi_class="ovo")
    print("[sklearn_metrics]Train Epoch:{} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} auc:{:.4f} SPE:{:.4f}".format(
        epoch, accu_loss.item() / (step + 1), sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1, sklearn_auc, sklearn_specificity))

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        #for i in range()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def patches_proposal(fold,root:str,img_train:str,task:str,patch_num:int,patch_size:int):
    img_list_file = open(img_train, "r")
    img_list = img_list_file.readlines()
    p_list = []
    n_list = []
    for i in img_list:
        path = os.path.join(root, i[:-1])
        if str(i)[0:2] == "AD":

            p_list.append(path)

        else:
            n_list.append(path)
    A = 0
    B = 0
    C = 0
    p_sum_ls = []
    n_sum_ls = []
    for P in p_list:
        img = nib.load(P).get_data()
        # 图片中不为零的区域：（18:162,18:197,0:154）
        # 28:152, 40:184, 9:153
        img = img[45:145, 56:181, 40:140]
        patch_step = patch_size // 2
        patches = patchify(img, (patch_size, patch_size, patch_size), step=patch_step)
        a, b, c, _, _, _ = patches.shape
        A = a
        B = b
        C = c
        sum_P = np.zeros((a, b, c))
        for i in range(a):
            for j in range(b):
                for h in range(c):
                    sum_P[i, j, h] = patches[i, j, h].sum()
        p_sum_ls.append(sum_P)

    for N in n_list:
        img = nib.load(N).get_data()#15:165, 15:200, 40:165
        img = img[45:145, 56:181, 40:140]
        patch_step = patch_size//2
        patches = patchify(img, (patch_size, patch_size, patch_size), step=patch_step)
        a, b, c, _, _, _ = patches.shape
        A = a
        B = b
        C = c
        sum_N = np.zeros((a, b, c))
        for i in range(a):
            for j in range(b):
                for h in range(c):
                    sum_N[i, j, h] = patches[i, j, h].sum()
        n_sum_ls.append(sum_N)

        # 得到正样本和副样本的每个位置的和矩阵
    print(len(p_sum_ls), len(n_sum_ls))
    p_sum_arr = np.array(p_sum_ls)
    n_sum_arr = np.array(n_sum_ls)
    print(p_sum_arr.shape, n_sum_arr.shape)

    loc_p_value = np.zeros((A, B, C))
    for i in range(A):
        for j in range(B):
            for h in range(C):
                loc_p_value[i, j, h] = ttest_ind(p_sum_arr[:, i, j, h], n_sum_arr[:, i, j, h]).pvalue
    # np.argmax()平铺数组 返回第一次出现的最大值的索引 一维的
    # np.unravel_index(A,(3,3))A为第几个数 此函数返回第几个数在原数组中的位置索引
    loc_p_value[np.isnan(loc_p_value)] = 1000
    df_p = pd.DataFrame(loc_p_value.reshape(A * B * C))
    df_p_60 = df_p.sort_values(by=0)[0:patch_num]
    df_loc_60 = df_p_60.index.values
    loc_patches_60 = []
    # 根据索引得到p值最小的前六十个块的位置在原数组中的索引
    for i in df_loc_60:
        loc_patches_60.append(np.unravel_index(i, (A, B, C)))

    df_loc_patches_60 = pd.DataFrame(loc_patches_60)
    df_loc_patches_60.to_csv("Loc_proposal/{}_location_proposal_{}_fold_{}_{}.csv".format(task,patch_size,fold,patch_num))

    return df_loc_patches_60

def patches_proposal_LARGE(fold,root:str,img_train:str,task:str,patch_num:int,patch_size:int):
    img_list_file = open(img_train, "r")
    img_list = img_list_file.readlines()
    p_list = []
    n_list = []
    for i in img_list:
        path = os.path.join(root, i[:-1])
        if str(i)[0:2] == "AD" or str(i)[0:3] == "MCI":

            p_list.append(path)

        else:
            n_list.append(path)
    A = 0
    B = 0
    C = 0
    p_sum_ls = []
    n_sum_ls = []
    for P in p_list:
        img = nib.load(P).get_data()
        # 图片中不为零的区域：（18:162,18:197,0:154）
        # 28:152, 40:184, 9:153
        img = img[36:154, 56:184, 40:140]
        patch_step = patch_size // 2
        patches = patchify(img, (patch_size, patch_size, patch_size), step=patch_step)
        a, b, c, _, _, _ = patches.shape
        A = a
        B = b
        C = c
        sum_P = np.zeros((a, b, c))
        for i in range(a):
            for j in range(b):
                for h in range(c):
                    sum_P[i, j, h] = patches[i, j, h].sum()
        p_sum_ls.append(sum_P)

    for N in n_list:
        img = nib.load(N).get_data()#15:165, 15:200, 40:165
        # 45:145, 56:181, 40:140
        img = img[36: 154, 56: 184, 40: 140]
        patch_step = patch_size//2
        patches = patchify(img, (patch_size, patch_size, patch_size), step=patch_step)
        a, b, c, _, _, _ = patches.shape
        A = a
        B = b
        C = c
        sum_N = np.zeros((a, b, c))
        for i in range(a):
            for j in range(b):
                for h in range(c):
                    sum_N[i, j, h] = patches[i, j, h].sum()
        n_sum_ls.append(sum_N)

        # 得到正样本和副样本的每个位置的和矩阵
    print(len(p_sum_ls), len(n_sum_ls))
    p_sum_arr = np.array(p_sum_ls)
    n_sum_arr = np.array(n_sum_ls)
    print(p_sum_arr.shape, n_sum_arr.shape)

    loc_p_value = np.zeros((A, B, C))
    for i in range(A):
        for j in range(B):
            for h in range(C):
                loc_p_value[i, j, h] = ttest_ind(p_sum_arr[:, i, j, h], n_sum_arr[:, i, j, h]).pvalue
    # np.argmax()平铺数组 返回第一次出现的最大值的索引 一维的
    # np.unravel_index(A,(3,3))A为第几个数 此函数返回第几个数在原数组中的位置索引
    loc_p_value[np.isnan(loc_p_value)] = 1000
    df_p = pd.DataFrame(loc_p_value.reshape(A * B * C))
    df_p_60 = df_p.sort_values(by=0)[0:patch_num]
    df_loc_60 = df_p_60.index.values
    loc_patches_60 = []
    # 根据索引得到p值最小的前六十个块的位置在原数组中的索引
    for i in df_loc_60:
        loc_patches_60.append(np.unravel_index(i, (A, B, C)))

    df_loc_patches_60 = pd.DataFrame(loc_patches_60)
    df_loc_patches_60.to_csv("Loc_proposal_Large/{}_location_proposal_{}_fold_{}_{}.csv".format(task,patch_size,fold,patch_num))

    return df_loc_patches_60



def patches_proposal_test(fold,patch_size,task,patch_num):
    df_loc_patches_60 = pd.read_csv("Loc_proposal/{}_location_proposal_{}_fold_{}_{}.csv".format(task,patch_size,fold,patch_num),header=0,index_col=0)
    return df_loc_patches_60



def patches_proposal_DAMIDL(fold,root:str,img_train:str,task:str,patch_num:int,patch_size:int):
    img_list_file = open(img_train, "r")
    img_list = img_list_file.readlines()
    p_list = []
    n_list = []
    for i in img_list:
        path = os.path.join(root, i[:-1])
        if str(i)[0:2] == "AD" or str(i)[0:4] == "PMCI":
            p_list.append(path)

        else:
            n_list.append(path)
    A = 0
    B = 0
    C = 0
    p_sum_ls = []
    n_sum_ls = []
    for P in p_list:
        img = nib.load(P).get_data()
        # 图片中不为零的区域：（18:162,18:197,0:154）
        # 28:152, 40:184, 9:153
        img = img[45:145, 56:181, 40:140]
        patch_step = patch_size
        patches = patchify(img, (patch_size, patch_size, patch_size), step=patch_step)
        a, b, c, _, _, _ = patches.shape
        A = a
        B = b
        C = c
        sum_P = np.zeros((a, b, c))
        for i in range(a):
            for j in range(b):
                for h in range(c):
                    sum_P[i, j, h] = patches[i, j, h].sum()
        p_sum_ls.append(sum_P)

    for N in n_list:
        img = nib.load(N).get_data()#15:165, 15:200, 40:165
        img = img[45:145, 56:181, 40:140]
        patch_step = patch_size
        patches = patchify(img, (patch_size, patch_size, patch_size), step=patch_step)
        a, b, c, _, _, _ = patches.shape
        A = a
        B = b
        C = c
        sum_N = np.zeros((a, b, c))
        for i in range(a):
            for j in range(b):
                for h in range(c):
                    sum_N[i, j, h] = patches[i, j, h].sum()
        n_sum_ls.append(sum_N)

        # 得到正样本和副样本的每个位置的和矩阵
    print(len(p_sum_ls), len(n_sum_ls))
    p_sum_arr = np.array(p_sum_ls)
    n_sum_arr = np.array(n_sum_ls)
    print(p_sum_arr.shape, n_sum_arr.shape)

    loc_p_value = np.zeros((A, B, C))
    for i in range(A):
        for j in range(B):
            for h in range(C):
                loc_p_value[i, j, h] = ttest_ind(p_sum_arr[:, i, j, h], n_sum_arr[:, i, j, h]).pvalue
    # np.argmax()平铺数组 返回第一次出现的最大值的索引 一维的
    # np.unravel_index(A,(3,3))A为第几个数 此函数返回第几个数在原数组中的位置索引
    loc_p_value[np.isnan(loc_p_value)] = 1000
    df_p = pd.DataFrame(loc_p_value.reshape(A * B * C))
    df_p_60 = df_p.sort_values(by=0)[0:patch_num]
    df_loc_60 = df_p_60.index.values
    loc_patches_60 = []
    # 根据索引得到p值最小的前六十个块的位置在原数组中的索引
    for i in df_loc_60:
        loc_patches_60.append(np.unravel_index(i, (A, B, C)))

    df_loc_patches_60 = pd.DataFrame(loc_patches_60)
    df_loc_patches_60.to_csv("Loc_proposal/DAMIDL_{}_location_proposal_fold_{}.csv".format(task,fold))

    return df_loc_patches_60
