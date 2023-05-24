import torch
import pandas as pd
from process import AudioUtil
from CNN import AudioClassifier
from loader import build_dataloader_test
from torch import nn
from utils import val
from unet import UNet


if __name__ == '__main__':
    model_path = '../output/fold/model_fold2_epoch19.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    read_data_file = '../input/2classes/test/test.csv'
    df = pd.read_csv(read_data_file)
    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    df = df[['relative_path', 'classID']]
    df = df.sample(frac=1).reset_index(drop=True)
    data_path = '../input/2classes/test'

    aud = []
    for index, row in df.iterrows():
        audList = AudioUtil.open_for_list(data_path + row['relative_path'])
        for i in audList:
            i.append(row['classID'])
            aud.append(i)

    # model = AudioClassifier()
    model = UNet(2)
    # print(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    test_dl = build_dataloader_test(aud)

    acc = val(model,test_dl,device)



