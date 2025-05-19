# PyTorch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as tr

# Models
from LRNet import LRNet
from LossFunction import HybridCDLoss

# Other
from thop import profile
from thop import clever_format
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from math import floor, ceil, sqrt, exp
from IPython import display
import time
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")

from Method.Utils.ChangeDetectionUtils import get_current_date,check_dir_or_create
from Method.Utils.SaveIndicatorsDuringTraining import save_indicators
from TOOL import *
from Parameters4LRNet import Parameters

def run(times):

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

    OUTPUT_RESULT_DIR = f"../../../../ResultRepo/{dict_dataset[TYPE_DATASET]}/LRNet_AVG_iter"
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
    criterion = HybridCDLoss(label_smoothing_para_beta=parameters.beta, hard_ratio_para_theta=parameters.theta)

    dummy_input = torch.randn((1, 3, 256, 256)).cuda()
    flops, params = profile(net, (dummy_input, dummy_input))
    flops, params = clever_format([flops, params], '%.2f')
    print('flops: ', flops, 'params: ', params)

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
        epoch_train_change_accuracy = 0 * t
        epoch_train_nochange_accuracy = 0 * t
        epoch_train_precision = 0 * t
        epoch_train_recall = 0 * t
        epoch_train_Fmeasure = 0 * t
        epoch_test_loss = 0 * t
        epoch_test_accuracy = 0 * t
        epoch_test_change_accuracy = 0 * t
        epoch_test_nochange_accuracy = 0 * t
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
            if parameters.TYPE_DATASET == 1:
                if epoch_index>15:
                    break
            net.train()
            print('\nEpoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS))

            epoch_index_loss = []
            epoch_index_total_loss = []
            for batch in tqdm(train_loader, position=0, desc=f"TRAIN EPOCH{epoch_index+1:>3}/{n_epochs}"):
                I1 = batch['I1'].float().cuda()
                I2 = batch['I2'].float().cuda()
                label = torch.squeeze(batch['label'].cuda())
                label_layer5 = torch.squeeze(batch['label32'].cuda())

                optimizer.zero_grad()
                output, output_layer5 = net(I1, I2)
                loss = criterion(output, label)
                loss_layer5 = criterion(output_layer5, label_layer5)
                total_loss = loss + loss_layer5
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
                epoch_train_loss[epoch_index], epoch_train_accuracy[epoch_index], cl_acc, pr_rec, k = test_with_dataloader(valid_loader)  # train_dataset
                epoch_train_nochange_accuracy[epoch_index] = cl_acc[0]
                epoch_train_change_accuracy[epoch_index] = cl_acc[1]
                epoch_train_precision[epoch_index] = pr_rec[0]
                epoch_train_recall[epoch_index] = pr_rec[1]
                epoch_train_Fmeasure[epoch_index] = pr_rec[2]

            epoch_test_loss[epoch_index], epoch_test_accuracy[epoch_index], cl_acc, pr_rec, k = test_with_dataloader(test_loader)#test_dataset
            epoch_test_nochange_accuracy[epoch_index] = cl_acc[0]
            epoch_test_change_accuracy[epoch_index] = cl_acc[1]
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

            plt.figure(num=3)
            plt.clf()
            l3_3, = plt.plot(t[:epoch_index + 1], epoch_test_nochange_accuracy[:epoch_index + 1],
                             label='Test accuracy: no change')
            l3_4, = plt.plot(t[:epoch_index + 1], epoch_test_change_accuracy[:epoch_index + 1],
                             label='Test accuracy: change')
            if VAL_OR_NOT == True:
                l3_1, = plt.plot(t[:epoch_index + 1], epoch_train_nochange_accuracy[:epoch_index + 1],
                                 label='Val accuracy: no change')
                l3_2, = plt.plot(t[:epoch_index + 1], epoch_train_change_accuracy[:epoch_index + 1],
                                 label='Val accuracy: change')
                plt.legend(handles=[l3_1, l3_2, l3_3, l3_4])
            else:
                plt.legend(handles=[l3_3, l3_4])
            plt.grid()
            plt.gcf().gca().set_ylim(0, 100)
            plt.title('Accuracy per class')
            display.clear_output(wait=True)
            display.display(plt.gcf())

            plt.figure(num=4)
            plt.clf()
            l4_4, = plt.plot(t[:epoch_index + 1], epoch_test_precision[:epoch_index + 1], label='Test precision')
            l4_5, = plt.plot(t[:epoch_index + 1], epoch_test_recall[:epoch_index + 1], label='Test recall')
            l4_6, = plt.plot(t[:epoch_index + 1], epoch_test_Fmeasure[:epoch_index + 1], label='Test Dice/F1')
            if VAL_OR_NOT == True:
                l4_1, = plt.plot(t[:epoch_index + 1], epoch_train_precision[:epoch_index + 1], label='Val precision')
                l4_2, = plt.plot(t[:epoch_index + 1], epoch_train_recall[:epoch_index + 1], label='Val recall')
                l4_3, = plt.plot(t[:epoch_index + 1], epoch_train_Fmeasure[:epoch_index + 1], label='Val Dice/F1')
                plt.legend(handles=[l4_1, l4_2, l4_3, l4_4, l4_5, l4_6])
            else:
                plt.legend(handles=[l4_4, l4_5, l4_6])
            plt.grid()
            plt.gcf().gca().set_ylim(0, 1)
            plt.title('Precision, Recall and F-measure')
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
                save_str_fm = OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-iter{times}-best_epoch-' + str(epoch_index + 1) + '_fm-' + str(round(fm, 8)) + '.pth.tar'
                torch.save(net.state_dict(), save_str_fm)
                temp_save_str_fm = save_str_fm
            else:
                save_str_fm = OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-iter{times}-epoch-' + str(epoch_index + 1) + '_fm-' + str(round(fm, 8)) + '.pth.tar'
                torch.save(net.state_dict(), save_str_fm)

            lss = epoch_train_loss[epoch_index]
            if VAL_OR_NOT == False:
                lss = epoch_test_loss[epoch_index]
            if lss < best_lss:
                best_lss = lss
                if temp_save_str_loss != "":
                    os.remove(temp_save_str_loss)
                save_str_loss = OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-iter{times}-best_epoch-' + str(epoch_index + 1) + '_loss-' + str(round(lss, 8)) + '.pth.tar'
                torch.save(net.state_dict(), save_str_loss)
                temp_save_str_loss = save_str_loss

            if save:
                im_format = 'png'
                # im_format = 'eps'

                plt.figure(num=1)
                plt.savefig(OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-iter{times}-loss.' + im_format)
                plt.figure(num=2)
                plt.savefig(OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-iter{times}-accuracy.' + im_format)
                plt.figure(num=3)
                plt.savefig(OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-iter{times}-accuracy-per-class.' + im_format)
                plt.figure(num=4)
                plt.savefig(OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-iter{times}-prec-rec-fmeas.' + im_format)
                plt.figure(num=5)
                plt.savefig(OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-iter{times}-LR.' + im_format)

                out = {'train_loss': epoch_train_loss[epoch_index],
                       'train_accuracy': epoch_train_accuracy[epoch_index],
                       'train_nochange_accuracy': epoch_train_nochange_accuracy[epoch_index],
                       'train_change_accuracy': epoch_train_change_accuracy[epoch_index],
                       'test_loss': epoch_test_loss[epoch_index],
                       'test_accuracy': epoch_test_accuracy[epoch_index],
                       'test_nochange_accuracy': epoch_test_nochange_accuracy[epoch_index],
                       'test_change_accuracy': epoch_test_change_accuracy[epoch_index]}
                print(str(out))

                print('pr_c, rec_c, f_meas, pr_nc, rec_nc')
                print(str(pr_rec))


        PATH_SAVE_INDICATORS = OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(
            TYPE_DATASET) + f'-{CUR_DATE}-iter{times}-Indicators.csv'
        indicators = {"epoch_loss": epoch_loss,
                      "epoch_total_loss": epoch_total_loss,
                      "epoch_LR": epoch_LR,
                      "epoch_train_loss": epoch_train_loss,
                      "epoch_train_accuracy": epoch_train_accuracy,
                      "epoch_train_change_accuracy": epoch_train_change_accuracy,
                      "epoch_train_nochange_accuracy": epoch_train_nochange_accuracy,
                      "epoch_train_precision": epoch_train_precision, "epoch_train_recall": epoch_train_recall,
                      "epoch_train_Fmeasure": epoch_train_Fmeasure, "epoch_test_loss": epoch_test_loss,
                      "epoch_test_accuracy": epoch_test_accuracy, "epoch_test_change_accuracy": epoch_test_change_accuracy,
                      "epoch_test_nochange_accuracy": epoch_test_nochange_accuracy,
                      "epoch_test_precision": epoch_test_precision, "epoch_test_recall": epoch_test_recall,
                      "epoch_test_Fmeasure": epoch_test_Fmeasure}
        save_indicators(PATH_SAVE=PATH_SAVE_INDICATORS, Indicators=indicators)

        return out


    def test_with_dataloader(dloader):
        net.eval()
        tot_loss = 0
        tot_count = 0

        n = 2
        class_correct = list(0. for i in range(n))
        class_total = list(0. for i in range(n))
        class_accuracy = list(0. for i in range(n))

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        with torch.no_grad():
            for batch in tqdm(dloader, position=0, desc="VAL/TEST"):
                I1 = batch['I1'].float().cuda()
                I2 = batch['I2'].float().cuda()
                cm = torch.squeeze(batch['label'].cuda())

                output = net(I1, I2)[0]

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

                class_correct[0] += tn
                class_correct[1] += tp
                class_total[0] += tn + fp
                class_total[1] += tp + fn

            net_loss = tot_loss / tot_count
            net_accuracy = torch.true_divide(100 * (tp + tn) , tot_count)

            for i in range(n):
                class_accuracy[i] = torch.true_divide(100 * class_correct[i] , max(class_total[i], 0.00001))

            prec = torch.true_divide(tp , (tp + fp))
            rec = torch.true_divide(tp , (tp + fn))
            f_meas = torch.true_divide(2 * prec * rec , (prec + rec))
            prec_nc = torch.true_divide(tn , (tn + fn))
            rec_nc = torch.true_divide(tn , (tn + fp))

            pr_rec = [prec, rec, f_meas, prec_nc, rec_nc]
            k = kappa(tp, tn, fp, fn)

            return net_loss, net_accuracy, class_accuracy, pr_rec, k


    check_dir_or_create(OUTPUT_RESULT_DIR)
    if LOAD_TRAINED:
        PATH_STATE_DICT = OUTPUT_RESULT_DIR + "/" + parameters.PATH_STATE_DICT
        net.load_state_dict(torch.load(PATH_STATE_DICT))
        print('LOAD OK')
    else:
        t_start = time.time()
        out_dic = train()
        t_end = time.time()
        print(out_dic)
        TimeOfTrain = round(t_end - t_start, 4)
        print('Elapsed time (Train): {}s'.format(TimeOfTrain))


    if not LOAD_TRAINED:
        PATH_STATE_DICT = OUTPUT_RESULT_DIR + "/" + net_name + '-DATA' + str(TYPE_DATASET) + f'-{CUR_DATE}-iter{times}_final.pth.tar'
        torch.save(net.state_dict(), PATH_STATE_DICT)
        print('SAVE OK')


    net_loss, net_accuracy, cl_acc, pr_rec, k =test_with_dataloader(test_loader)
    results = {'net_loss': round(float(net_loss.cpu().numpy()),6),
               'net_accuracy': round(float(net_accuracy.cpu().numpy()),6),
               'class_accuracy': round(float(cl_acc[1].cpu().numpy()),6),
               'class_accuracy_nc': round(float(cl_acc[0].cpu().numpy()),6),
               'precision': round(float(pr_rec[0].cpu().numpy()),6),
               'recall': round(float(pr_rec[1].cpu().numpy()),6),
               'f-means': round(float(pr_rec[2].cpu().numpy()),6),
               'precision_nc': round(float(pr_rec[3].cpu().numpy()),6),
               'recall_nc': round(float(pr_rec[4].cpu().numpy()),6),
               'kappa': round(float(k.cpu().numpy()),6)}
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
            # "N_VAL":valid_dataset.n_imgs,
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
                I1, I2, cm = dset.get_img(name)
                I1 = torch.unsqueeze(I1, 0).float().cuda()
                I2 = torch.unsqueeze(I2, 0).float().cuda()
                out = net(I1, I2)[0]
                predicted = torch.round(out.data)
                predicted = np.squeeze(predicted.cpu().numpy())
                I = np.stack((255 * cm, 255 * predicted, 255 * cm), 2).astype(np.uint8)
                io.imsave(PATH_SAVE_RESULT + f'{net_name}-{dict_dataset[TYPE_DATASET]}-{name}.png', I)
                I_Predict = np.stack((255 * predicted, 255 * predicted, 255 * predicted), 2).astype(np.uint8)
                io.imsave(PATH_SAVE_RESULT + f'{net_name}-{dict_dataset[TYPE_DATASET]}-{name}-Pred.png', I_Predict)

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

if __name__ == '__main__':
    time_run=5
    for i in range(1,time_run+1):
        print(f'----------ITER  {i}/{time_run}  开始----------')
        run(times=i)
        print(f'----------ITER  {i}/{time_run}  完成----------')
    print(f"运行{time_run}次完成!!!")