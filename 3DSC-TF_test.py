import os
import torch
from visualizer import get_local
from torch.utils.data import DataLoader
get_local.activate()
from model import VisionTransformer as create_model
from read_nii import MyDataSet_3D_60_patches as MYDATASET


from sklearn import metrics
from sklearn.metrics import confusion_matrix
from utils import read_split_data_tvt_5cv_test, patches_proposal_test

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
fold = 0
patch_num = 60
patch_size = 32
image_test = "./ADNC/test_{}.csv".format(fold)
image_train = "./ADNC/train_{}.csv".format(fold)
task = "ADNC"
root = "/home/disk1/zhouqinyi/SMALLDATASET/ADNC"

test_images_path, test_images_label = read_split_data_tvt_5cv_test(image_test, root)
model_path = "./.pth"

model = create_model(embed_dim=96, slice_num=60,
                     num_classes=2,depth=4,num_heads=8,
                     mlp_ratio=2,drop_ratio=0.5,
                     attn_drop_ratio=0.5,
                     drop_path_ratio=0.5).to(device)
patches_pick = patches_proposal_test(fold = fold, patch_size = patch_size, task = task, patch_num=patch_num)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set model to evaluate mode

test_dataset = MYDATASET(images_path=test_images_path,
                         images_class=test_images_label,patches_loc = patches_pick,
                                  patch_size=patch_size, patch_num=patch_num)

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
    correct = 0
    total = 0
    for step, data in enumerate(test_loader):
        image0, labels = data
        sample_num += image0.shape[0]
        outputs = model.forward(image0.to(device))

        _, predicted = torch.max(outputs.data, 1)
        predict = torch.softmax(outputs, dim=1)[:, 1]
        # print(predict,labels.to(device),test_images_path[step])
        if predicted != labels.to(device):
            wronglist.append(test_images_path[step])
            print(predict, labels.to(device), test_images_path[step])
            wrongpath = "./wrongpredict"
            if labels.to(device) == 0:
                wrongpath_NC = os.path.join(wrongpath,test_images_path[step][-17:])

            else:
                wrongpath_AD = os.path.join(wrongpath, test_images_path[step][-17:])


        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()
        test_preds.extend(predicted.detach().cpu().numpy())
        test_trues.extend(labels.detach().cpu().numpy())
        pred_roc.extend(predict.detach().cpu().numpy())
    confusion = confusion_matrix(test_trues, test_preds)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    sklearn_specificity = TN / float(TN + FP)
    test_acc = correct / total
    sklearn_precision = TP / float(TP + FP)
    sklearn_recall = TP / float(TP + FN)
    sklearn_f1 = 2 * sklearn_recall * sklearn_precision / (sklearn_recall + sklearn_precision)
    sklearn_auc = metrics.roc_auc_score(test_trues, pred_roc, multi_class="ovo")

    print(
        "[sklearn_metrics] fold{} Test accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} auc:{:.4f} SPE:{:.4f}".format(
            fold,
            test_acc,
            sklearn_precision,
            sklearn_recall,
            sklearn_f1,
            sklearn_auc,
            sklearn_specificity))
    print('fold{} Test accuracy of the model on the test images: {} %'
          .format(fold, 100 * test_acc))
