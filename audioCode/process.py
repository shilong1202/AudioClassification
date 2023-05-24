import math, random
import torch
import torchaudio
from matplotlib import pyplot as plt

# print(torchaudio.list_audio_backends())
torchaudio.set_audio_backend("soundfile")  # 切换到 libsox 后端
from torchaudio import transforms
from IPython.display import Audio


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


if __name__=='__main__':

    audio_file = "E:\softSpace\PycharmSpaces\pytorch37\\audioClassification\input\\26-29_09_2017_KCL\ReadText\HC\ID00_hc_0_0_0.wav"
    segment_length = 30  # 30 seconds

    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file)

    # Calculate the length of each segment in samples
    samples_per_segment = sample_rate * segment_length
    print(samples_per_segment)
    print(sample_rate)
    print(waveform.shape)

    # Split the audio waveform into segments
    # exit()
    segmented_waveforms = []
    for i in range(0, waveform.shape[-1], samples_per_segment):
        segment = waveform[..., i:i + samples_per_segment]
        if segment.shape[-1] == samples_per_segment:
            segmented_waveforms.append(segment)

    print(len(segmented_waveforms))
    # To use each segmented waveform, treat it just like any other audio waveform tensor:
    # for segment in segmented_waveforms:
    #     # Do something with the segment, for example:
    #     pass


#     audioUtil= AudioUtil()
#     # 读取音频文件
#     sig,st = audioUtil.open('E:\softSpace\PycharmSpaces\pytorch37\\audioClassification\input\\26-29_09_2017_KCL\ReadText\HC\ID00_hc_0_0_0.wav') # sig : torch.Size([1, 6664159])
#     # print(sig.shape)
#     # 将转换为立体声
#     resig, sr = audioUtil.rechannel((sig,st),2)
#     # print(resig.shape)
#     # 标准化采样率
#     resig, newsr = audioUtil.resample((resig, sr),44100)
#     # print(resig.shape)
#     # print(newsr)
#     # 填充
#     resig, newsr = audioUtil.pad_trunc((resig, newsr),150)
#     # print(resig.shape)
#     # print(newsr)
#
#     # 数据扩充增广：时移
#     # resig, newsr = audioUtil.time_shift((resig, newsr),150)
#
#     spec = audioUtil.spectro_gram((resig, newsr),hop_len=40)
#     # print(spec.shape)
#     # print(type(spec))
#
#     audioUtil.show_spectro_gram(spec)
#
#     # 数据扩充：时间和频率屏蔽
#     spec = audioUtil.spectro_augment(spec)
#
#     audioUtil.show_spectro_gram(spec)