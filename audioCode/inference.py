import random
import torch
import torchaudio
from torchaudio import transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
torchaudio.set_audio_backend("soundfile")  # 切换到 libsox 后端



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels - out_channels, in_channels - out_channels, kernel_size=2,
                                                stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        cut = x.shape[3] if x.shape[3] <= skip_input.shape[3] else skip_input.shape[3]
        x = torch.cat([x, skip_input[:,:,:,:cut]], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_classes=2, up_sample_mode='bilinear'):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Down sampling Path
        self.down_conv1 = DownBlock(in_ch, 8)
        self.down_conv2 = DownBlock(8, 16)
        self.down_conv3 = DownBlock(16, 32)
        self.down_conv4 = DownBlock(32, 64)

        # Bottleneck
        self.double_conv = DoubleConv(64, 128)
        # Sampling Path
        self.up_conv4 = UpBlock(64 + 128, 64, self.up_sample_mode)
        self.up_conv3 = UpBlock(32 + 64, 32, self.up_sample_mode)
        self.up_conv2 = UpBlock(16 + 32, 16, self.up_sample_mode)
        self.up_conv1 = UpBlock(24, 32, self.up_sample_mode)

        # Final Convolution
        self.conv_last = nn.Conv2d(32, 64, kernel_size=1)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=2)


    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)
        return x


class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # 读取文件中的音频
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        # print(type(sig))
        return (sig, sr)

    @staticmethod
    def openList(audio_file):
        segment_length = 10  # 30 seconds
        overlap_length = 5  # 5 seconds

        waveform, sample_rate = torchaudio.load(audio_file)

        samples_per_segment = sample_rate * segment_length
        samples_per_overlap = sample_rate * overlap_length

        segmented_waveforms = []
        for i in range(sample_rate * 30, waveform.shape[-1] - samples_per_overlap, samples_per_segment - samples_per_overlap):
            segment = waveform[..., i:i + samples_per_segment]
            segmented_waveforms.append([segment,sample_rate])
        return segmented_waveforms

    @staticmethod
    def open_for_list(audio_file):
        segment_length = 10  # 30 seconds
        overlap_length = 5  # 5 seconds

        waveform, sample_rate = torchaudio.load(audio_file)

        samples_per_segment = sample_rate * segment_length
        samples_per_overlap = sample_rate * overlap_length

        segmented_waveforms = []
        for i in range(sample_rate * 30, waveform.shape[-1] - samples_per_overlap, samples_per_segment - samples_per_overlap):
            segment = waveform[..., i:i + samples_per_segment]
            segmented_waveforms.append([segment,sample_rate])
        return segmented_waveforms
    # ----------------------------
    # Convert the given audio to the desired number of channels
    # 转换成立体声
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return resig, sr

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # 标准化采样率
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        if (sr == newsr):
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # 将所有音频样本的大小调整为具有相同的长度,使用静默填充或通过截断其长度来延长其持续时间
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms
        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # 数据扩充增广：时移
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    # ----------------------------
    # Generate a Spectrogram
    # 梅尔频谱图
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def show_spectro_gram(spec):
        mel_specgram = spec.numpy()

        plt.imshow(mel_specgram[0], cmap='jet', origin='lower', aspect='auto')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel bins')
        plt.title('Mel Spectrogram')
        plt.show()

    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # 数据扩充：时间和频率屏蔽
    # ----------------------------

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec


def infer(audio_path):
    model_path = './model_fold2_epoch19.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audList = AudioUtil.open_for_list(audio_path)
    model = UNet(2)
    # print(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    inp = None
    for i in audList:
        # print(i[0].shape)
        reaud = AudioUtil.resample((i[0], i[1]), 44100)
        # print(reaud[0].shape)
        # exit()
        rechan = AudioUtil.rechannel(reaud, 2)

        dur_aud = AudioUtil.pad_trunc(rechan, 10000)
        # print(dur_aud[0].shape)
        shift_aud = AudioUtil.time_shift(dur_aud, 0.4)
        # print(shift_aud[0].shape)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        # print(sgram.shape)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        # print(aug_sgram.shape)
        aug_sgram=aug_sgram.to(device)
        # print(aug_sgram.shape)

        inputs = aug_sgram.unsqueeze(0)
        if inp is None:
            inp = inputs
        else:
            inp = torch.cat((inp,inputs),dim=0)

        # Normalize the inputs
    inputs_m, inputs_s = inp.mean(), inp.std()
    inputs = (inp - inputs_m) / inputs_s
    outputs = model(inputs)
    _, prediction = torch.max(outputs, 1)

    log_softmax_tensor = F.softmax(outputs, dim=1)

    # print(log_softmax_tensor)
    # 生成PD类型的概率
    mean = log_softmax_tensor[:, 0].mean()

    arr = mean.cpu().detach().numpy()
    return arr



if __name__ == '__main__':
    audio_path = 'E:/softSpace/PycharmSpaces/pytorch37/audioClassification/input/2classes/test/fold2/IDSp07_pd_2_0_0.wav'
    res = infer(audio_path)
    print(res)
