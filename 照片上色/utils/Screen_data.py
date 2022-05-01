import os

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image

def get_fileName(filePath):
    return os.path.splitext(os.path.basename(filePath))[0]

def create_data(img_size,path_pic='./data/train1/*.jpg'):

    data_transform = transforms.Compose([
                                transforms.Scale([img_size, img_size]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ])

    data_transform_Gray = transforms.Compose([                                
                                transforms.Grayscale(num_output_channels=1), 
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ])

    train_data = {}
    label_data = {}
    fileNameList = glob.glob(path_pic)
    k = 0
    #实现构建训练集与测试集，主要是将原来的彩色图片转化为黑白图片，之后以黑白图片为训练集
    #以原始的彩色图片为目标进行训练
    for i, filename in enumerate(fileNameList):
        img_color1 = Image.open(filename).convert('RGB')

        img_color2 = data_transform(img_color1)
        if img_color2.shape[0] == 3:

            img = img_color1.convert('L')
            img = img.resize((img_size, img_size), Image.BICUBIC)
            img = data_transform_Gray(img)
            train_data[k] = img
            label_data[k] = img_color2
            k+=1
    return train_data, label_data

#构建dataloader数据集
class prepare_Dataset(Dataset):
    def __init__(self, df_train, df_label):
        self.df_train = df_train
        self.df_label = df_label


    def __getitem__(self, index):
        data0 = self.df_train[index]
        data_label = self.df_label[index]
        return data0, data_label
    
    def __len__(self):
        return len(self.df_train)