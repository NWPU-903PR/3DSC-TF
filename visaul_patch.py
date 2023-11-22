from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch
from visualizer import get_local
import numpy as np
import pandas as pd
# from vit_model import vit_base_patch16_224_in21k as create_model
get_local.activate()
from model.Transformer_3D import VisionTransformer as create_model
from my_dataset import MyDataSet_3D_60_patches_Large as MYDATASET
import os
import shutil
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils import read_split_data_tvt_5cv_test,patches_proposal_test
import seaborn as sns

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
fold = 4
patch_num = 60
patch_size = 32
image_test = ./ADNC/test_{}.csv".format(fold)
image_train = ./ADNC/train_{}.csv".format(fold)
task = "ADNC"
# image_test = "/home/disk1/zhouqinyi/PycharmProjects/originvit/datasplit/exp1/exp1test.csv"
root = "./ADNC"

test_images_path, test_images_label = read_split_data_tvt_5cv_test(image_test, root)
model_path = "./.pth".format(fold)

model = create_model(embed_dim=96, slice_num=60,
                     num_classes=2,depth=4,num_heads=8,
                     mlp_ratio=2,drop_ratio=0.5,
                     attn_drop_ratio=0.5,
                     drop_path_ratio=0.5).to(device)
patches_pick = patches_proposal_test(fold = fold, patch_size = patch_size, task = task,patch_num=60)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set model to evaluate mode

test_dataset = MYDATASET(images_path=test_images_path,
                         images_class=test_images_label,patches_loc = patches_pick,
                                  patch_size=patch_size, patch_num=patch_num)
attention_map_mean_list = []
attention_map_list=[]
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=8,
                                          collate_fn=test_dataset.collate_fn)
test_preds = []
pred_roc = []
test_trues = []
wronglist = []
sample_num = 0
# Test the model
model.eval()  # eval mode(batch norm uses moving mean/variance
# instead of mini-batch mean/variance)

with torch.no_grad():
    for step, data in enumerate(test_loader):
        image0, labels = data
        sample_num += image0.shape[0]
        get_local.clear()
        outputs = model.forward(image0.to(device))
        _, predicted = torch.max(outputs.data, 1)
        predict = torch.softmax(outputs, dim=1)[:, 1]
        # print(predict,labels.to(device),test_images_path[step])
        if predicted == labels.to(device) and labels.to(device)==1:
            cache = get_local.cache
            attention_map = cache['Attention.forward']
            # 得到所有层的attmap
            # att_map = attention_map[0].squeeze()
            att_array = np.array(attention_map)#(4,1,8,61,61)
            attention_map_list.append(att_array)
            # 得到所有层的attmap每个头平均
            att_map_mean = att_array.mean(axis=2)
            attention_map_mean_list.append(att_map_mean)
            # print(list(cache.keys()))
        elif predicted != labels.to(device) and labels.to(device)==1:
            print(test_images_path[step])

att_map_cls = np.array(attention_map_list)
# att_map_mean_cls = np.array(attention_map_mean_list)
# att_map_mean = att_map_mean_cls.mean(axis=0)
att_mean = att_map_cls.mean(axis=0)
c = att_mean.mean(axis = 2)
d = c.squeeze()
x1 = d[0]
x2 = d[1]
x3 = d[2]
x4 = d[3]
y1 = np.dot(x1,x2)
y2 = np.dot(y1,x3)
y3 = np.dot(y2,x4)
att_score = y3[0,1:]
att = pd.DataFrame(y3[0,1:],index = [f'slcie_{i}' for i in range(60)], columns = ["attention_score"])
att.insert(loc = 1,column = "slice_num",value = [f'{i}' for i in range(60)])
max_min = y3[0,1:].max()-y3[0,1:].min()
att_score_norm = []
for i in range(60):
    a = att_score[i]-att_score.min()
    att_score_norm.append(a/max_min)
print(len(att_score_norm))
att.insert(loc = 2,column = "att_score",value = att_score_norm)
sns.barplot(x = "slice_num",y = "att_score",data = att)
att.to_csv("AD_att_FOLD_{}.csv".format(fold))
plt.show()
