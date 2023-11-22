import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import numpy as np
import sys
import math
import time
from tqdm import tqdm
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from read_nii import MyDataSet_3D_60_patches as MYDATASET
from model import VisionTransformer as create_model
from utils import read_split_data_tvt_5cv,patches_proposal
from sklearn import metrics
from sklearn.metrics import precision_score, f1_score, confusion_matrix

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):


        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def main(args):
    test_5cv = []
    task = args.task
    patch_num = args.patch_num
    patch_size = args.patch_size
    model_name = args.task_model
    embed_dim = args.embed_dim
    lr_patience = args.lr_patience
    ES_patience = args.ES_patience
    for i in range(0,5):
        print("fold_{}".format(i))
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        if task == "ADNC":
            image_train = "./ADNC/train_{}.csv".format(i)
            image_test = "./ADNC/test_{}.csv".format(i)
        elif task == "MCINC":
            image_train = "./MCINC/train_{}.csv".format(i)
            image_test = "./MCINC/test_{}.csv".format(i)
        elif task == "ADMCI":
            image_train = "./ADMCI/train_{}.csv".format(i)
            image_test = "./ADMCI/test_{}.csv".format(i)

        if os.path.exists("./.") is False:
            os.makedirs("./.")
        patches_pick = patches_proposal(i,args.data_path,image_train,task,patch_num,patch_size)

        tb_writer = SummaryWriter()

        train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data_tvt_5cv(image_train, image_test,
            args.data_path)
        print("Number of positive samples in the training set：", train_images_label.count(1), "Number of negative samples in the training set：", train_images_label.count(0))
        print("Number of positive samples in the validation set：", val_images_label.count(1), "Number of negative samples in the validation set：", val_images_label.count(0))
        print("Number of positive samples in the test set：", test_images_label.count(1), "Number of negative samples in the test set：", test_images_label.count(0))


        train_dataset = MYDATASET(images_path=train_images_path,
                                  images_class=train_images_label,patches_loc = patches_pick,
                                  patch_size=patch_size, patch_num=patch_num)

        val_dataset = MYDATASET(images_path=val_images_path,
                                images_class=val_images_label,patches_loc = patches_pick,
                                patch_size = patch_size, patch_num = patch_num)
        test_dataset = MYDATASET(images_path=test_images_path,
                                 images_class=test_images_label,patches_loc = patches_pick,
                                 patch_size=patch_size, patch_num=patch_num)

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=2,
                                                  collate_fn=test_dataset.collate_fn)

        model = create_model(embed_dim=embed_dim, slice_num=patch_num,
                             num_classes=2,depth=4,num_heads=8,mlp_ratio=2,drop_ratio=0,attn_drop_ratio=0,drop_path_ratio=0).to(device)

        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        early_stopping = EarlyStopping(patience=ES_patience, verbose=True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=lr_patience,verbose=True)

        acc = 0
        bestepoch = 0
        prec = 0
        recall = 0
        f1 = 0
        auc = 0
        spe = 0

        loss_list = []
        acc_list = []
        val_loss_list = []
        val_acc_list = []

        for epoch in range(args.epochs):

            model.train()
            loss_function = torch.nn.CrossEntropyLoss()
            accu_loss = torch.zeros(1).to(device)
            accu_num = torch.zeros(1).to(device)
            optimizer.zero_grad()

            sample_num = 0
            train_loader = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_loader):
                image, labels = data
                sample_num += image.shape[0]
                pred = model.forward(image.to(device))
                pred_classes = torch.max(pred, dim=1)[1]
                accu_num += torch.eq(pred_classes, labels.to(device)).sum()
                loss = loss_function(pred, labels.to(device))
                loss.backward()
                accu_loss += loss.detach()
                train_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                        accu_loss.item() / (step + 1),
                                                                                        accu_num.item() / sample_num)
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss)
                    sys.exit(1)

                optimizer.step()
                optimizer.zero_grad()
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            train_loss, train_acc = accu_loss.item() / (step + 1), accu_num.item() / sample_num
            loss_list.append(train_loss)
            acc_list.append(train_acc)


            loss_function = torch.nn.CrossEntropyLoss()
            model.eval()
            accu_num = torch.zeros(1).to(device)
            accu_loss = torch.zeros(1).to(device)
            sample_num = 0
            val_loader = tqdm(val_loader, file=sys.stdout)

            with torch.no_grad():
                for step, data in enumerate(val_loader):
                    image, labels = data
                    sample_num += image.shape[0]
                    pred = model.forward(image.to(device))
                    pred_classes = torch.max(pred, dim=1)[1]
                    accu_num += torch.eq(pred_classes, labels.to(device)).sum()
                    # for i in range()
                    loss = loss_function(pred, labels.to(device))
                    accu_loss += loss
                    val_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                          accu_loss.item() / (step + 1),
                                                                                          accu_num.item() / sample_num)
                val_loss, val_acc = accu_loss.item() / (step + 1), accu_num.item() / sample_num
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)
            scheduler.step(val_loss)
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)


            test_preds = []
            test_trues = []
            pred_roc = []
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for step, data in enumerate(test_loader):
                    image, labels = data
                    sample_num += image.shape[0]
                    outputs = model.forward(image.to(device))

                    _, predicted = torch.max(outputs.data, 1)
                    predict = torch.softmax(outputs, dim=1)[:, 1]

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
                sklearn_precision = precision_score(test_trues, test_preds, average='weighted')
                sklearn_recall = TP / float(TP + FN)
                sklearn_f1 = f1_score(test_trues, test_preds, average='weighted')
                sklearn_auc = metrics.roc_auc_score(test_trues, pred_roc, multi_class="ovo")

                print(
                    "[sklearn_metrics] Test accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} auc:{:.4f} SPE:{:.4f}".format(
                        test_acc,
                        sklearn_precision,
                        sklearn_recall,
                        sklearn_f1,
                        sklearn_auc,
                        sklearn_specificity))

                print('Test accuracy of the model on the test images: {} %'
                      .format(100 * test_acc))

            if (acc < val_acc) or (acc == val_acc and besttest < test_acc):
                acc = val_acc
                besttest = test_acc
                prec = sklearn_precision
                recall = sklearn_recall
                f1 = sklearn_f1
                auc = sklearn_auc
                spe = sklearn_specificity

                bestepoch = epoch
                torch.save(model.state_dict(), "./{}_{}_fold_{}best.pth".format(model_name,task,i))

                print(
                    'Best VAL epoch TEST: {:.0f} VALacc: {:.4f} TESTacc: {:.4f} PREC: {:.4f} SEN(RECALL): {:.4f} F1: {:.4f} AUC: {:.4f} SPE:{:.4f}'
                    .format(bestepoch, acc, besttest, prec, recall, f1, auc, spe))

        print(
            'fold{} Best epoch: {:.0f}  VALacc: {:.4f} TESTacc: {:.4f} PREC: {:.4f} SEN(RECALL): {:.4f} F1: {:.4f} AUC: {:.4f} SPE:{:.4f}'
            .format(i, bestepoch, acc, besttest, prec, recall, f1, auc, spe))
        test_5cv.append('fold{} Best epoch: {:.0f}  VALacc: {:.4f} TESTacc: {:.4f} PREC: {:.4f} SEN(RECALL): {:.4f} F1: {:.4f} AUC: {:.4f} SPE: {:.4f}'
            .format(i, bestepoch, acc, besttest, prec, recall, f1, auc, spe))

    resultfile = open("Results/{}/{}_{}_result.csv".format(task,model_name,task), "w")
    for i in test_5cv:
        resultfile.write(i + "\n")

if __name__ == '__main__':

    old_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--lr_patience', type=float, default=8)
    parser.add_argument('--ES_patience', type=float, default=50)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--task', default='ADNC')
    parser.add_argument('--task_model', default="TF4CNN+1+2DSC_32_3")
    parser.add_argument('--patch_num', default=60)
    parser.add_argument('--embed_dim', default=96)
    parser.add_argument('--patch_size', default=32)


    parser.add_argument('--data-path', type=str,
                        default="/home/disk1/zhouqinyi/SMALLDATASET/ADNC")
    parser.add_argument('--model-name', default='', help='create model name')

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=False)

    opt = parser.parse_args()


    main(opt)
    current_time = time.time()
    print("The running time is" + str(current_time - old_time) + "s")