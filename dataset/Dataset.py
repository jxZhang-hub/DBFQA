import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision
import cv2 as cv

unloader = torchvision.transforms.ToPILImage()

def make_gradeint(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    gradxy = torch.from_numpy(gradxy)
    gradxy = gradxy.permute(2,0,1)
    gradxy = gradxy.cpu().clone()
    gradxy = gradxy.squeeze(0)
    gradxy = unloader(gradxy)
    gradxy = gradxy.convert('L')
    gradxy = np.array(gradxy)
    return gradxy

#reading image as the grayscale format
def loader(path):
    img = Image.open(path)
    img = img.convert('L')
    ret = img.copy()
    img.close()
    ret = np.array(ret)
    return ret

#reading lines in .txt file
def lineget(path):
    f = open(path, 'r')
    ret = []
    for line in f.readlines():
        ret.append(line.strip())
    return ret

def label_Normalize(labellist):
    label = []
    #str -> float
    for i in range(len(labellist)):
        labellist[i] = eval(labellist[i])


    min = np.min(np.array(labellist))
    max = np.max(np.array(labellist))
    for i in range(len(labellist)):
        label.append((labellist[i] - min)/(max - min))

    return label


class IQADataset(Dataset):
    def __init__(self, index):
        self.im_names = lineget("D:\Project\FQA\DBFQA\dataset\im_names.txt")
        self.mos = label_Normalize(lineget("D:\Project\FQA\DBFQA\dataset\mos.txt"))

        self.img_names = []
        self.label = []

        for idx in index:
            self.img_names.append(self.im_names[idx])
            self.label.append(self.mos[idx])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        label = self.label[idx]
        img_gray = loader(os.path.join("./data/data/", img_name))
        img_gray = cv.resize(img_gray, (224, 224))
        img_gray = img_gray / 255.0

        '''
        img_gradient = cv.imread(os.path.join("./data/data_e/", img_name))
        img_gradient = make_gradeint(img_gradient)
        img_gradient = cv.resize(img_gradient, (224, 224))
        img_gradient = img_gradient / 255.0
        '''

        img_edge = loader(os.path.join("./data/data_e", img_name))
        img_edge = cv.resize(img_edge, (224, 224))
        img_edge = img_edge / 255.0


        #replace the "img_edge" into "img_gradient" if necessary
        return (torch.tensor(img_gray), torch.tensor(img_edge)), torch.tensor(label).float()


#testing Dataset.py
if __name__ == '__main__':
    idx = list(range(10000))
    #random.shuffle(idx)
    train_idx = idx[:int(0.8 * len(idx))]

    train_set = IQADataset(train_idx)

    (img, img_e), label = train_set.__getitem__(0)
    print(img.shape)
    print(img_e.shape)
    print(label)
    print("Success")
