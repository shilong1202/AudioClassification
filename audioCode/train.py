import matplotlib.pyplot as plt

from loader import build_dataloader, build_dataloader_fold
import pandas as pd
from CNN import AudioClassifier
from unet import UNet
import torch
from utils import training
from sklearn.model_selection import StratifiedGroupKFold, KFold
from process import AudioUtil

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 20
    lr = 0.001
    n_fold = 2
    bs = 1

    read_data_file = '../input/2classes/sound.csv'
    df = pd.read_csv(read_data_file)
    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    df = df[['relative_path', 'classID']]
    df = df.sample(frac=1).reset_index(drop=True)
    data_path = '../input/2classes'

    aud = []
    for index, row in df.iterrows():
        audList = AudioUtil.openList(data_path + row['relative_path'])
        for i in audList:
            i.append(row['classID'])
            aud.append(i)

    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(aud)):  # flod= 0、1、2、3
        for i in val_idx:
            aud[i].append(fold)

    show_acc = []
    show_loss = []
    best_val_acc_list = []
    for fold in range(n_fold):
        print(f'#' * 40, flush=True)
        print(f'###### Fold: {fold}', flush=True)
        print(f'#' * 40, flush=True)

        train_dl, val_dl = build_dataloader_fold(aud, fold, bs)

        # model = AudioClassifier()
        model = UNet(2)
        # print(model)
        model = model.to(device)

        best_val_acc, loss_list, acc_list = training(model, train_dl, num_epochs, device, lr, val_dl, fold)
        show_loss.append(loss_list)
        show_acc.append(acc_list)
        best_val_acc_list.append(best_val_acc)

    epoch_size_list = [i + 1 for i in range(num_epochs)]
    fig = plt.Figure()
    plt.subplot(1, 2, 1)
    plt.plot(epoch_size_list, show_acc[0], c='blue', label='acc')
    plt.plot(epoch_size_list, show_loss[0], c='red', label='loss')
    plt.title(' Fold:' + str(1) + 'val loss:' + str(best_val_acc_list[0]))

    plt.subplot(1, 2, 2)
    plt.plot(epoch_size_list, show_acc[1], c='blue', label='acc')
    plt.plot(epoch_size_list, show_loss[1], c='red', label='loss')
    plt.title(' Fold:' + str(2) + 'val loss:' + str(best_val_acc_list[1]))

    # plt.subplot(2, 2, 3)
    # plt.plot(epoch_size_list, show_acc[2], c='blue', label='acc')
    # plt.plot(epoch_size_list, show_loss[2], c='red', label='loss')
    # plt.title(' Fold:' + str(3) + 'val loss:' + str(best_val_acc_list[2]))
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(epoch_size_list, show_acc[3], c='blue', label='acc')
    # plt.plot(epoch_size_list, show_loss[3], c='red', label='loss')
    # plt.title(' Fold:' + str(4) + 'val loss:' + str(best_val_acc_list[3]))

    plt.show()
