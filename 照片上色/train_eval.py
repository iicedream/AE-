import pandas as pd
from torch.utils.data import DataLoader
import torch
import numpy as np
def train_(ae, data_train, optimizer, mse_loss, device, batch_size, n_epochs, sample_interval, save_img_1):
    # ----------
    #  Training
    # ----------
    train_datas = DataLoader(dataset=data_train,batch_size=batch_size, shuffle=True, drop_last=False)

    losses_g = []
    losses_d = []
    for epoch in range(n_epochs):
        epoch += 1
        gl = []
        dl = []
        for i, data in enumerate(train_datas):
            
            imgs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            validity_real = ae(imgs)
            #每隔50个epoch保存图片
            if ((epoch == 1) and (i ==1)) or ((epoch % 50 ==0) and (i ==1)):
                save_img_1(labels, imgs ,validity_real, epoch, Train=True)

            loss = mse_loss(validity_real, labels)
            loss.backward()

            optimizer.step()
            gl.append(loss.item())

        #每隔sample_interval次保存一次模型
        if epoch % sample_interval == 0:
            torch.save(ae.state_dict(), './model/AE_epoch_%d.pth' % (epoch))

        losses_g.append(np.mean(gl))

        print ("[Epoch %d/%d] [loss: %f]" % (epoch, n_epochs, loss.item()))

    losses_pd = pd.DataFrame(data=losses_g)
    losses_pd.to_csv('loss.csv')
    return losses_pd

def test_(ae, test_data, device, save_img_1):
    # ----------
    #  Test
    # ----------
    ae.eval()
    test_datas = DataLoader(dataset=test_data,batch_size=1, shuffle=False, drop_last=False)

    for i, data in enumerate(test_datas):
            
        imgs = data[0]
        labels = data[1]

        real_imgs = imgs.to(device)
        labels = labels.to(device)

        validity_real = ae(real_imgs)
        save_img_1(labels, real_imgs,validity_real, i)

        if i % 50 ==0:
            print(i)