from process import AudioUtil
from unet import UNet
import torch
import numpy as np

def infer(audio_path, device, model_path):
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
    # print(outputs)

    # log_softmax_tensor  = F.softmax(outputs, dim=1)
    # print(log_softmax_tensor)
    #
    # sum_0 = 0
    # sum_1 = 0
    # for i in log_softmax_tensor:
    #     sum_0 += i[0]
    #     sum_1 += i[1]
    #     # print(i[0])
    #     # predicted_class = torch.argmax(i)
    #     # print(predicted_class)
    # print(sum_0)
    # print(sum_1)

    _, prediction = torch.max(outputs, 1)
    # print(prediction)

    # 将张量转换为 NumPy 数组
    arr = prediction.cpu().numpy()

    # 统计 1 和 0 的个数
    num_ones = np.count_nonzero(arr)
    num_zeros = arr.size - num_ones

    # 输出结果
    # print("Number of ones: {}".format(num_ones))
    # print("Number of zeros: {}".format(num_zeros))

    if num_ones>num_zeros:
        print("音频为HC类型音频")
    else:
        print("音频为PD类型音频")


if __name__ == '__main__':
    # model_path = '../output/fold/model_10s_fold1_epoch24.pth'
    model_path = '../output/fold/model_fold2_epoch19.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audio_path = 'E:/softSpace/PycharmSpaces/pytorch37/audioClassification/input/2classes/test/fold1/IDSp03_hc_0_0_0.wav'
    # audio_path = 'E:/softSpace/PycharmSpaces/pytorch37/audioClassification/input/2classes/test/fold2/IDSp04_pd_2_0_1.wav'
    # audio_path = 'E:/softSpace/PycharmSpaces/pytorch37/audioClassification/input/2classes/fold2/ID02_pd_2_0_0.wav'
    # audio_path = 'E:/softSpace/PycharmSpaces/pytorch37/audioClassification/input/2classes/fold1/ID00_hc_0_0_0.wav'
    infer(audio_path, device, model_path)
