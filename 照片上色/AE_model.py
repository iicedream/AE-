from numpy import block
import torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
            #编码块
        def encoder_block(in_filters, out_filters, kernel_size, stride, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, 0)]
            block.append(nn.ReLU())
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
            #解码块
        def dencoder_block(in_filters, out_filters,kernel_size,stride, bn=True):
            block = [nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride, 0)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.append(nn.Tanh())
            return block

        self.model = nn.Sequential(
            *encoder_block(1, 32, 5, 3),
            *encoder_block(32, 32, 3, 1),
            *encoder_block(32, 32, 3, 1),

            *dencoder_block(32, 32, 3, 1),
            *dencoder_block(32, 32, 3, 1),
            *dencoder_block(32, 3, 5, 3),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
