from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from process import AudioUtil
from torch.utils.data import random_split
import torch


# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

        # ----------------------------

    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'classID']

        aud = AudioUtil.open(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id


def build_dataloader(df, data_path):
    # myds = SoundDS(df, data_path)
    myds = SoundDS(df, data_path)
    # Random split of 80:20 between training and validation
    # print(type(myds))
    num_items = len(myds)
    print(num_items)
    exit()
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_dl, val_dl


class SoundDSSplice(Dataset):
    def __init__(self, aud_list):
        self.aud_list = aud_list
        # self.data_path = str(data_path)
        self.duration = 10000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    def __len__(self):
        return len(self.aud_list)

    def __getitem__(self, idx):
        aud = self.aud_list[idx][0]
        reaud = AudioUtil.resample((aud, self.aud_list[idx][1]), self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, self.aud_list[idx][2]


def build_dataloader_fold(aud, fold , bs):
    aud_train = []
    aud_val = []
    for a in aud:
        if a[3] != fold:
            aud_train.append(a)
        else:
            aud_val.append(a)

    data_train = SoundDSSplice(aud_train)
    data_val = SoundDSSplice(aud_val)

    print('feature number：' + str(len(data_train)+len(data_val)))

    train_dl = torch.utils.data.DataLoader(data_train, batch_size=bs, shuffle=True)
    val_dl = torch.utils.data.DataLoader(data_val, batch_size=bs, shuffle=True)

    return train_dl, val_dl


def build_dataloader_test(aud):
    data_test = SoundDSSplice(aud)

    print('feature number：' + str(len(data_test)))

    test_dl = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=True)

    return test_dl