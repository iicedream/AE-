import numpy as np
import matplotlib.pyplot as plt

from utils.Screen_data import create_data, prepare_Dataset

def read_Dataset(fileName,img_size):
    data_train, lable_train = create_data(img_size,fileName)
    data_train = prepare_Dataset(data_train,lable_train)
    return data_train

def tensor_to_img(image):
    #dataLoader中设置的mean与std参数
    mean = [0.5,0.5,0.5] 
    std = [0.5,0.5,0.5]

    image_tensor = image.data
    image_numpy = image_tensor.cpu().float().numpy()
    #转为RGB并且反标准化
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    for i in range(len(mean)): 
        image_numpy[i] = image_numpy[i] * std[i] + mean[i]
    #反ToTensor(),从[0,1]转为[0,255]
    image_numpy = image_numpy * 255
    # 从(通道, 高, 宽)转化为(高, 宽, 通道)从而满足输入特征
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    result = image_numpy.astype(np.uint8)

    return result

def save_img_1(img_tensor_label,image_tensor_gry,image_tensor_gen,epoch,Train=False):
    #训练时保存图片
    if Train:
        fig = plt.figure(figsize = (10, 10))
        rows = 8
        cols = 3
        k=1
        for i in range(1, rows + 1):
            for ek, img_tensor in enumerate([img_tensor_label,image_tensor_gry,image_tensor_gen]):
                img_tensor = img_tensor[i-1,:,:,:]
                # 子图位置
                ax = fig.add_subplot(rows, cols, k)
                plt.axis('off')
                image = img_tensor.cpu().detach().numpy()
                image = tensor_to_img(img_tensor)
                if ek == 1:
                    plt.imshow(image,cmap=plt.get_cmap('gray'))
                else:
                    plt.imshow(image)
                k += 1
        plt.savefig('./picture/%d.jpg' %epoch)

    #测试时保存图片
    else:
        fig = plt.figure()
        rows = 1
        cols = 3
        k=1
        for i in range(1, rows + 1):
            for ek, img_tensor in enumerate([img_tensor_label,image_tensor_gry,image_tensor_gen]):
                img_tensor = img_tensor[i-1,:,:,:]

                ax = fig.add_subplot(rows, cols, k)
                plt.axis('off')
                image = img_tensor.cpu().detach().numpy()
                image = tensor_to_img(img_tensor)
                if ek == 1:
                    plt.imshow(image,cmap=plt.get_cmap('gray'))
                else:
                    plt.imshow(image)
                k += 1
        plt.savefig('./score/{}_{}.jpg'.format(epoch,ek),dpi=2000)