import argparse
import os

import torch
import matplotlib.pyplot as plt

from AE_model import AE
from utils.tools import read_Dataset, save_img_1
from train_eval import train_, test_

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
parser.add_argument('--sample_interval', type=int, default=50, help='interval between image sampling')
parser.add_argument('--Train', type=bool, default=False, help='does it require training')
parser.add_argument('--train_data_path', type=str, default='./data/train/*.jpg', help='train data path')
parser.add_argument('--test_data_path', type=str, default='./data/test/*.jpg', help='test data path')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义损失函数
mse_loss = torch.nn.MSELoss().to(device)

ae = AE().to(device)
# Optimizers
optimizer = torch.optim.Adam(ae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


if opt.Train:
    train_data = read_Dataset(opt.train_data_path, opt.img_size)
    loss = train_(ae, train_data,  optimizer, mse_loss, device, opt.batch_size, opt.n_epochs, opt.sample_interval, save_img_1)
    plt.figure()
    plt.plot(loss,label = 'loss')

    plt.legend()
    plt.savefig('loss.jpg')
test_data = read_Dataset(opt.test_data_path,opt.img_size)

ae = AE().to(device)
ae.load_state_dict(torch.load(os.path.join('./model/', "AE_epoch_{}.pth".format(opt.n_epochs))))
test_(ae, test_data, device, save_img_1)