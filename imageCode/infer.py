import cv2
import torch
import torchvision
from torch.autograd import Variable
import numpy as np
import warnings
warnings.simplefilter("ignore", category=UserWarning)
import torch.nn.functional as F

def infer(path):
    model1_vgg16 = torchvision.models.vgg16(pretrained=False)
    num_fc = model1_vgg16.classifier[6].in_features
    model1_vgg16.classifier[6] = torch.nn.Linear(num_fc, 2)
    for param in model1_vgg16.parameters():
        param.requires_grad = False
    model1_vgg16.load_state_dict(torch.load('vgg.pth'))
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = (torch.from_numpy(img))
    img_tensor = img_tensor.unsqueeze(0)
    model1_vgg16.eval()
    batch_x = Variable(img_tensor)
    out = model1_vgg16(batch_x)
    log_softmax_tensor = F.softmax(out)
    return log_softmax_tensor.cpu().detach().numpy()[0][1]


if __name__ =='__main__':
    res = infer('E:\softSpace\PycharmSpaces\pytorch37\\audioClassification\input\Spirals\\testing\HC\V06HE01.png')
    print(res)