# PyTorch
import numpy as np
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as tr

# Models
from LRNet import LRNet
from LossFunction import HybridCDLoss, DiceLoss, HybridCDLoss_IOU, IOULoss_Edge

# # Other
# from thop import profile
# from thop import clever_format
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from math import floor, ceil, sqrt, exp
from IPython import display
import time
import cv2 as cv
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")

from Method.Utils.ChangeDetectionUtils import get_current_date,check_dir_or_create
from Method.Utils.SaveIndicatorsDuringTraining import save_indicators
from TOOL4Edge import *
from Parameters4LRNetEdge import Parameters
parameters=Parameters()
cuda_device_id=parameters.cuda_device_id
print("torch version       ：",torch.__version__)
print("torch cuda available：",torch.cuda.is_available())
print("torch device count  ：",torch.cuda.device_count())
torch.cuda.set_device(cuda_device_id)
print("torch device current：",torch.cuda.current_device())
print('IMPORTS OK')


# Global Variables
TYPE_DATASET = parameters.TYPE_DATASET # 1:DSIFN-Dataset | 2:WHU-Building-Dataset | 3:LEVIR-CD | 4:S2Looking | 5:WHU-BCD
dict_dataset = {0: "TEST", 1: "DSIFN-Dataset", 2: "WHU-Building-Dataset", 3: "LEVIR-CD", 4: "S2Looking", 5: "WHU-BCD"}
N_CHANNEL = 3
PATH_DATASET = f'../../../../DataRepo/{dict_dataset[TYPE_DATASET]}/'
PATH_TRAIN_DATASET = PATH_DATASET + 'train/'
PATH_VAL_DATASET = PATH_DATASET + 'val/'
PATH_TEST_DATASET = PATH_DATASET + 'test/'
NORMALISE_IMGS = parameters.NORMALISE_IMGS
MODE_NORMALISE = parameters.MODE_NORMALISE
DATA_AUG = parameters.DATA_AUG
VAL_OR_NOT = parameters.ValOrNot

N_EPOCHS = parameters.N_EPOCHS   #50
BATCH_SIZE = parameters.BATCH_SIZE
PATCH_SIDE = parameters.PATCH_SIDE #128 512
LOAD_TRAINED = parameters.LOAD_TRAINED

OUTPUT_RESULT_DIR = f"../../../../ResultRepo/{dict_dataset[TYPE_DATASET]}/LRNet_Edge"
CUR_DATE=get_current_date()

print("---------Parameters---------")
print(str(dict(parameters.__dict__)))
print('DEFINITIONS OK')


# Dataset
if DATA_AUG:
    data_transform = tr.Compose([RandomFlip(), RandomRot()])
else:
    data_transform = None

train_dataset = ChangeDetectionDataset(TYPE_DATASET, path=PATH_TRAIN_DATASET, train_val_test ='train', patch_side = PATCH_SIDE, NORMALISE=NORMALISE_IMGS, MODE_NORMALISE=MODE_NORMALISE, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0, pin_memory=True)   #num_workers = 4
test_dataset = ChangeDetectionDataset(TYPE_DATASET, path=PATH_TEST_DATASET, train_val_test ='test', patch_side = PATCH_SIDE, NORMALISE=NORMALISE_IMGS, MODE_NORMALISE=MODE_NORMALISE, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0, pin_memory=True)   #num_workers = 4
if VAL_OR_NOT==True:
    valid_dataset = ChangeDetectionDataset(TYPE_DATASET, path=PATH_VAL_DATASET, train_val_test ='val', patch_side = PATCH_SIDE, NORMALISE=NORMALISE_IMGS, MODE_NORMALISE=MODE_NORMALISE, transform=data_transform)
    valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0, pin_memory=True)    #num_workers = 4

print(f'DATASETS {dict_dataset[TYPE_DATASET]} OK')


# NETWORK
net, net_name = LRNet(lrnet_cos_sim_threshold=parameters.lrnet_cos_sim_threshold, lrnet_label_threshold=parameters.lrnet_label_threshold), 'LRNet'
# print(net)

net.cuda()
criterion = HybridCDLoss_IOU(label_smoothing_para_beta=parameters.beta, hard_ratio_para_theta=parameters.theta)
criterion_edge = IOULoss_Edge()

# dummy_input = torch.randn((1, 3, 256, 256)).cuda()
# flops, params = profile(net, (dummy_input, dummy_input))
# flops, params = clever_format([flops, params], '%.2f')
# print('flops: ', flops, 'params: ', params)

print('NETWORK ' + net_name + ' OK')
NumOfTrainableParameters=count_parameters(net)
print('Number of trainable parameters:', NumOfTrainableParameters)


def train(n_epochs=N_EPOCHS, save=True):
    t = np.linspace(1, n_epochs, n_epochs)

    epoch_loss = 0 * t
    epoch_total_loss = 0 * t
    epoch_LR = 0 * t

    epoch_train_loss = 0 * t
    epoch_train_accuracy = 0 * t
    epoch_train_iou = 0 * t
    epoch_train_precision = 0 * t
    epoch_train_recall = 0 * t
    epoch_train_Fmeasure = 0 * t
    epoch_test_loss = 0 * t
    epoch_test_accuracy = 0 * t
    epoch_test_iou = 0 * t
    epoch_test_precision = 0 * t
    epoch_test_recall = 0 * t
    epoch_test_Fmeasure = 0 * t

    best_fm = 0
    best_lss = 1000

    plt.figure(num=1)
    plt.figure(num=2)
    plt.figure(num=3)
    plt.figure(num=4)
    plt.figure(num=5)

    optimizer = torch.optim.Adam(net.parameters(), lr=parameters.INIT_LR, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    temp_save_str_fm = ""
    temp_save_str_loss = ""
    for epoch_index in tqdm(range(n_epochs)):
        net.train()
        print('\nEpoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS))

        epoch_index_loss = []
        epoch_index_total_loss = []
        for batch in tqdm(train_loader, position=0, desc=f"TRAIN EPOCH{epoch_index+1:>3}/{n_epochs}"):
            I1 = batch['I1'].float().cuda()
            I2 = batch['I2'].float().cuda()
            label = torch.squeeze(batch['label'].cuda())
            label_layer5 = torch.squeeze(batch['label32'].cuda())
            edge = torch.squeeze(batch['edge'].cuda())
            edge32 = torch.squeeze(batch['edge32'].cuda())

            optimizer.zero_grad()
            output, output_layer5 = net(I1, I2)

            output_edge = torch.Tensor(transformEdge(input_img=torch.round(output.data), dilate=True)).float().cuda()
            output_edge32 = torch.Tensor(transformEdge(input_img=torch.round(output_layer5.data), dilate=False)).float().cuda()
            loss = criterion(output, label)
            loss_layer5 = criterion(output_layer5, label_layer5)
            loss_egde = criterion_edge(output_edge, edge)
            loss_egde32 = criterion_edge(output_edge32, edge32)
            total_loss = loss + loss_layer5 + loss_egde + loss_egde32
            total_loss.backward()
            optimizer.step()

            del batch

            epoch_index_loss.append(loss)
            epoch_index_total_loss.append(total_loss)

        epoch_loss[epoch_index] = torch.mean(torch.Tensor(epoch_index_loss))
        epoch_total_loss[epoch_index] = torch.mean(torch.Tensor(epoch_index_total_loss))
        epoch_LR[epoch_index] = optimizer.param_groups[0]["lr"]

        scheduler.step()

        if VAL_OR_NOT == True:
            epoch_train_loss[epoch_index], epoch_train_accuracy[epoch_index], pr_rec_edge, pr_rec, k = test_with_dataloader(valid_loader)  # train_dataset: net_loss, net_accuracy, pr_rec_edge, pr_rec_area, k
            epoch_train_iou[epoch_index] = pr_rec[3]  #pr_rec = [prec_chg, rec_chg, f_meas_chg, iou_chg, prec_nc, rec_nc]
            epoch_train_precision[epoch_index] = pr_rec[0]
            epoch_train_recall[epoch_index] = pr_rec[1]
            epoch_train_Fmeasure[epoch_index] = pr_rec[2]

        epoch_test_loss[epoch_index], epoch_test_accuracy[epoch_index], pr_rec_edge, pr_rec, k = test_with_dataloader(test_loader)#test_dataset: net_loss, net_accuracy, pr_rec_edge, pr_rec_area, k
        epoch_test_iou[epoch_index] = pr_rec[3]
        epoch_test_precision[epoch_index] = pr_rec[0]
        epoch_test_recall[epoch_index] = pr_rec[1]
        epoch_test_Fmeasure[epoch_index] = pr_rec[2]

        plt.figure(num=1)
        plt.clf()
        l1_2, = plt.plot(t[:epoch_index + 1], epoch_test_loss[:epoch_index + 1], label='Test loss')
        l1_3, = plt.plot(t[:epoch_index + 1], epoch_loss[:epoch_index + 1], label='Train loss')
        l1_4, = plt.plot(t[:epoch_index + 1], epoch_total_loss[:epoch_index + 1], label='Total loss')
        if VAL_OR_NOT == True:
            l1_1, = plt.plot(t[:epoch_index + 1], epoch_train_loss[:epoch_index + 1], label='Val loss')
            plt.legend(handles=[l1_1, l1_2, l1_3, l1_4])
        else:
            plt.legend(handles=[l1_2, l1_3, l1_4])
        plt.grid()
        plt.title('Loss')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=2)
        plt.clf()
        l2_2, = plt.plot(t[:epoch_index + 1], epoch_test_accuracy[:epoch_index + 1], label='Test accuracy')
        if VAL_OR_NOT == True:
            l2_1, = plt.plot(t[:epoch_index + 1], epoch_train_accuracy[:epoch_index + 1], label='Val accuracy')
            plt.legend(handles=[l2_1, l2_2])
        else:
            plt.legend(handles=[l2_2])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 100)
        plt.title('Accuracy')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=4)
        plt.clf()
        l4_4, = plt.plot(t[:epoch_index + 1], epoch_test_precision[:epoch_index + 1], label='Test precision')
        l4_5, = plt.plot(t[:epoch_index + 1], epoch_test_recall[:epoch_index + 1], label='Test recall')
        l4_6, = plt.plot(t[:epoch_index + 1], epoch_test_Fmeasure[:epoch_index + 1], label='Test F1')
        l4_7, = plt.plot(t[:epoch_index + 1], epoch_test_iou[:epoch_index + 1], label='Test IOU')
        if VAL_OR_NOT == True:
            l4_1, = plt.plot(t[:epoch_index + 1], epoch_train_precision[:epoch_index + 1], label='Val precision')
            l4_2, = plt.plot(t[:epoch_index + 1], epoch_train_recall[:epoch_index + 1], label='Val recall')
            l4_3, = plt.plot(t[:epoch_index + 1], epoch_train_Fmeasure[:epoch_index + 1], label='Val F1')
            l4_8, = plt.plot(t[:epoch_index + 1], epoch_train_Fmeasure[:epoch_index + 1], label='Val IOU')
            plt.legend(handles=[l4_1, l4_2, l4_3, l4_4, l4_5, l4_6, l4_7, l4_8])
        else:
            plt.legend(handles=[l4_4, l4_5, l4_6, l4_7])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 1)
        plt.title('Precision, Recall, F1 and IOU')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=5)
        plt.clf()
        l5_1, = plt.plot(t[:epoch_index + 1], epoch_LR[:epoch_index + 1], label='Learning Rate')
        plt.legend(handles=[l5_1])
        plt.grid()
        plt.title('Learning Rate')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        fm = epoch_test_Fmeasure[epoch_index]
        if VAL_OR_NOT == False:
            fm = epoch_test_Fmeasure[epoch_index]
        if fm > best_fm:
            best_fm = fm
            if temp_save_str_fm != "":
                os.remove(temp_save_str_fm)
            save_str_fm = OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-best_epoch-' + str(epoch_index + 1) + '_fm-' + str(round(fm, 8)) + '.pth.tar'
            torch.save(net.state_dict(), save_str_fm)
            temp_save_str_fm = save_str_fm
        else:
            save_str_fm = OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-epoch-' + str(epoch_index + 1) + '_fm-' + str(round(fm, 8)) + '.pth.tar'
            torch.save(net.state_dict(), save_str_fm)

        lss = epoch_train_loss[epoch_index]
        if VAL_OR_NOT == False:
            lss = epoch_test_loss[epoch_index]
        if lss < best_lss:
            best_lss = lss
            if temp_save_str_loss != "":
                os.remove(temp_save_str_loss)
            save_str_loss = OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-best_epoch-' + str(epoch_index + 1) + '_loss-' + str(round(lss, 8)) + '.pth.tar'
            torch.save(net.state_dict(), save_str_loss)
            temp_save_str_loss = save_str_loss

        if save:
            im_format = 'png'
            # im_format = 'eps'

            plt.figure(num=1)
            plt.savefig(OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-loss.' + im_format)
            plt.figure(num=2)
            plt.savefig(OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-accuracy.' + im_format)
            plt.figure(num=4)
            plt.savefig(OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-prec-rec-fmeas.' + im_format)
            plt.figure(num=5)
            plt.savefig(OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-LR.' + im_format)

            out = {'train_loss': epoch_train_loss[epoch_index],
                   'train_accuracy': epoch_train_accuracy[epoch_index],
                   'test_loss': epoch_test_loss[epoch_index],
                   'test_accuracy': epoch_test_accuracy[epoch_index]}
            print(str(out))

            print('Area: prec_chg, rec_chg, f_meas_chg, iou_chg, prec_nc, rec_nc') #pr_rec = [prec_chg, rec_chg, f_meas_chg, iou_chg, prec_nc, rec_nc]
            print(str(pr_rec))
            print('Edge: prec_edge_chg, rec_edge_chg, f_meas_edge_chg, iou_edge_chg, prec_edge_nc, rec_edge_nc')  # pr_rec_edge = [prec_edge_chg, rec_edge_chg, f_meas_edge_chg, iou_edge_chg, prec_edge_nc, rec_edge_nc]
            print(str(pr_rec_edge))


    PATH_SAVE_INDICATORS = OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(
        TYPE_DATASET) + f'-{CUR_DATE}-Indicators.csv'
    indicators = {"epoch_loss": epoch_loss,
                  "epoch_total_loss": epoch_total_loss,
                  "epoch_LR": epoch_LR,
                  "epoch_train_loss": epoch_train_loss,
                  "epoch_train_accuracy": epoch_train_accuracy,
                  "epoch_train_iou": epoch_train_iou,
                  "epoch_train_precision": epoch_train_precision, "epoch_train_recall": epoch_train_recall,
                  "epoch_train_Fmeasure": epoch_train_Fmeasure,
                  "epoch_test_loss": epoch_test_loss,
                  "epoch_test_accuracy": epoch_test_accuracy, "epoch_test_iou": epoch_test_iou,
                  "epoch_test_precision": epoch_test_precision, "epoch_test_recall": epoch_test_recall,
                  "epoch_test_Fmeasure": epoch_test_Fmeasure}
    save_indicators(PATH_SAVE=PATH_SAVE_INDICATORS, Indicators=indicators)

    return out


def test_with_dataloader(dloader):
    net.eval()
    tot_loss = 0
    tot_count = 0

    n = 2

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    tp_edge = 0
    tn_edge = 0
    fp_edge = 0
    fn_edge = 0

    with torch.no_grad():
        for batch in tqdm(dloader, position=0, desc="VAL/TEST"):
            I1 = batch['I1'].float().cuda()
            I2 = batch['I2'].float().cuda()
            cm = torch.squeeze(batch['label'].cuda())
            # label_layer5 = torch.squeeze(batch['label32'].cuda())
            edge = torch.squeeze(batch['edge'].cuda())
            # edge32 = torch.squeeze(batch['edge32'].cuda())

            output, output_layer5 = net(I1, I2)

            output_edge = torch.Tensor(transformEdge(input_img=torch.round(output.data), dilate=True)).float().cuda()
            # output_edge32 = torch.Tensor(transformEdge(input_img=torch.round(output_layer5.data), dilate=False)).float().cuda()

            loss = criterion(output, cm)
            tot_loss += loss.data * np.prod(cm.size())
            tot_count += np.prod(cm.size())

            predicted = torch.squeeze(torch.round(output.data))

            pr = predicted.int() > 0
            gt = cm.data.int() > 0

            tp += torch.logical_and(pr, gt).sum()
            tn += torch.logical_and(torch.logical_not(pr), torch.logical_not(gt)).sum()
            fp += torch.logical_and(pr, torch.logical_not(gt)).sum()
            fn += torch.logical_and(torch.logical_not(pr), gt).sum()

            pr_edge = torch.squeeze(output_edge).int() > 0
            gt_edge = torch.squeeze(edge).data.int() > 0

            tp_edge += torch.logical_and(pr_edge, gt_edge).sum()
            tn_edge += torch.logical_and(torch.logical_not(pr_edge), torch.logical_not(gt_edge)).sum()
            fp_edge += torch.logical_and(pr_edge, torch.logical_not(gt_edge)).sum()
            fn_edge += torch.logical_and(torch.logical_not(pr_edge), gt_edge).sum()


        net_loss = tot_loss / tot_count
        net_accuracy = torch.true_divide(100 * (tp + tn) , tot_count)

        prec_chg = torch.true_divide(tp , (tp + fp))
        rec_chg = torch.true_divide(tp , (tp + fn))
        f_meas_chg = torch.true_divide(2 * prec_chg * rec_chg , (prec_chg + rec_chg))
        prec_nc = torch.true_divide(tn , (tn + fn))
        rec_nc = torch.true_divide(tn , (tn + fp))
        iou_chg = torch.true_divide(tp , (tp + fn + fp))

        prec_edge_chg = torch.true_divide(tp_edge, (tp_edge + fp_edge))
        rec_edge_chg = torch.true_divide(tp_edge, (tp_edge + fn_edge))
        f_meas_edge_chg = torch.true_divide(2 * prec_edge_chg * rec_edge_chg, (prec_edge_chg + rec_edge_chg))
        prec_edge_nc = torch.true_divide(tn_edge, (tn_edge + fn_edge))
        rec_edge_nc = torch.true_divide(tn_edge, (tn_edge + fp_edge))
        iou_edge_chg = torch.true_divide(tp_edge, (tp_edge + fn_edge + fp_edge))


        pr_rec_area = [prec_chg, rec_chg, f_meas_chg, iou_chg, prec_nc, rec_nc]
        pr_rec_edge = [prec_edge_chg, rec_edge_chg, f_meas_edge_chg, iou_edge_chg, prec_edge_nc, rec_edge_nc]
        k = kappa(tp, tn, fp, fn)

        return net_loss, net_accuracy, pr_rec_edge, pr_rec_area, k


check_dir_or_create(OUTPUT_RESULT_DIR)
if LOAD_TRAINED:
    PATH_STATE_DICT = OUTPUT_RESULT_DIR + "/" + parameters.PATH_STATE_DICT
    net.load_state_dict(torch.load(PATH_STATE_DICT), strict=False)
    print('LOAD OK')
else:
    t_start = time.time()
    out_dic = train()
    t_end = time.time()
    print(out_dic)
    TimeOfTrain = round(t_end - t_start, 4)
    print('Elapsed time (Train): {}s'.format(TimeOfTrain))


if not LOAD_TRAINED:
    PATH_STATE_DICT = OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}_final.pth.tar'
    torch.save(net.state_dict(), PATH_STATE_DICT)
    print('SAVE OK')


net_loss, net_accuracy, pr_rec_edge, pr_rec_area, k =test_with_dataloader(test_loader)  #pr_rec = [prec_chg, rec_chg, f_meas_chg, iou_chg, prec_nc, rec_nc]
results = {'net_loss': round(float(net_loss.cpu().numpy()),4),
           'net_accuracy': round(float(net_accuracy.cpu().numpy()),2),
           'precision': round(float(pr_rec_area[0].cpu().numpy()),4),
           'recall': round(float(pr_rec_area[1].cpu().numpy()),4),
           'f-means': round(float(pr_rec_area[2].cpu().numpy()),4),
           'iou': round(float(pr_rec_area[3].cpu().numpy()),4),
           'precision_nc': round(float(pr_rec_area[4].cpu().numpy()),4),
           'recall_nc': round(float(pr_rec_area[5].cpu().numpy()),4),
           'precision_edge': round(float(pr_rec_edge[0].cpu().numpy()),4),
           'recall_edge': round(float(pr_rec_edge[1].cpu().numpy()),4),
           'f-means_edge': round(float(pr_rec_edge[2].cpu().numpy()),4),
           'iou_edge': round(float(pr_rec_edge[3].cpu().numpy()),4),
           'kappa': round(float(k.cpu().numpy()),4)}
pprint(results)

if LOAD_TRAINED == False:
    parameters2file={
        "N_EPOCHS":N_EPOCHS,
        "PATCH_SIZE":PATCH_SIDE,
        "BATCH_SIZE":BATCH_SIZE,
        "Num_Trainable_Parameters":NumOfTrainableParameters,
        "Time_Train":TimeOfTrain,
        "N_TRAIN":train_dataset.n_imgs,
        "N_TEST":test_dataset.n_imgs,
    }
    if VAL_OR_NOT == True:
        parameters2file["N_VAL"]=valid_dataset.n_imgs

    with open(f"{OUTPUT_RESULT_DIR}/result_output.txt","a") as f:
        date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write("\n")
        f.write("\n" + net_name + '-DATA' + str(TYPE_DATASET) + f"-{date_time}")
        f.write("\n")
        f.write(str(dict(parameters.__dict__)))
        f.write("\n")
        f.write(str(parameters2file))
        f.write("\n")
        f.write(str(results))
else:
    def save_test_results(dset):
        for name in tqdm(dset.names, position=0, desc="SAVE RESULT"):
            I1, I2, cm, cm32, edge, edge32 = dset.get_img(name) #img1, img2, cm, cm32, edge, edge32
            I1 = torch.unsqueeze(I1, 0).float().cuda()
            I2 = torch.unsqueeze(I2, 0).float().cuda()
            out = net(I1, I2)[0]
            output_edge = np.squeeze(transformEdge(input_img=torch.round(out.data), dilate=True))

            predicted = torch.round(out.data)
            predicted = np.squeeze(predicted.cpu().numpy())

            test_sample_id = parameters.test_id[parameters.TYPE_DATASET]

            I = np.stack((255 * cm, 255 * predicted, 255 * cm), 2).astype(np.uint8)
            I_Predict = np.stack((255 * predicted, 255 * predicted, 255 * predicted), 2).astype(np.uint8)
            E = np.stack((255 * edge, output_edge, 255 * edge), 2).astype(np.uint8)
            E_Predict = np.stack((output_edge, output_edge, output_edge), 2).astype(np.uint8)
            if name in test_sample_id:
                io.imsave(PATH_SAVE_RESULT + f'0-{net_name}-{dict_dataset[TYPE_DATASET]}-{name}.png', I)
                io.imsave(PATH_SAVE_RESULT + f'0-{net_name}-{dict_dataset[TYPE_DATASET]}-{name}-Pred.png', I_Predict)
                io.imsave(PATH_SAVE_RESULT + f'0-{net_name}-{dict_dataset[TYPE_DATASET]}-{name}-edge.png', E)
                io.imsave(PATH_SAVE_RESULT + f'0-{net_name}-{dict_dataset[TYPE_DATASET]}-{name}-edge-Pred.png', E_Predict)
            else:
                io.imsave(PATH_SAVE_RESULT + f'{net_name}-{dict_dataset[TYPE_DATASET]}-{name}.png', I)
                io.imsave(PATH_SAVE_RESULT + f'{net_name}-{dict_dataset[TYPE_DATASET]}-{name}-Pred.png', I_Predict)
                io.imsave(PATH_SAVE_RESULT + f'{net_name}-{dict_dataset[TYPE_DATASET]}-{name}-edge.png', E)
                io.imsave(PATH_SAVE_RESULT + f'{net_name}-{dict_dataset[TYPE_DATASET]}-{name}-edge-Pred.png', E_Predict)

    t_start = time.time()
    PATH_SAVE_RESULT = f"{OUTPUT_RESULT_DIR}/pred-{net_name}-{dict_dataset[TYPE_DATASET]}-{CUR_DATE}/"
    check_dir_or_create(PATH_SAVE_RESULT)
    save_test_results(test_dataset)
    t_end = time.time()
    print('Elapsed time (Save Result): {}s'.format(round(t_end - t_start, 4)))

    with open(f"{PATH_SAVE_RESULT}/0-result_output.txt", "a") as f:
        date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write("\n")
        f.write("\n" + f"{net_name}-{dict_dataset[TYPE_DATASET]}-{date_time}")
        f.write("\n")
        f.write(str(results))