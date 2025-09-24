import torch
import torch.nn as nn
from module import *

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = encoder_block(block=conv_block,in_channels=3, num_filters=64)
        self.encoder2 = encoder_block(block=conv_block,in_channels=64, num_filters=128)
        self.encoder3 = encoder_block(block=conv_block,in_channels=128, num_filters=256)
        self.encoder4 = encoder_block(block=conv_block,in_channels=256, num_filters=512)
        self.bridge = conv_block(in_channels=512, num_filters=1024)
        self.dencoder1 = decoder_block(block=conv_block, num_filters=512)
        self.dencoder2 = decoder_block(block=conv_block, num_filters=256)
        self.dencoder3 = decoder_block(block=conv_block, num_filters=128)
        self.dencoder4 = decoder_block(block=conv_block, num_filters=64)
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same')
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)
        b1 = self.bridge(p4)
        d1 = self.dencoder1(b1, s4)
        d2 = self.dencoder2(d1, s3)
        d3 = self.dencoder3(d2, s2)
        d4 = self.dencoder4(d3, s1)
        output = self.conv(d4)
        return self.sigmoid(output)

class UnetWithSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = encoder_block(block=conv_block, in_channels=1, num_filters=64)
        self.encoder2 = encoder_block(block=conv_block, in_channels=64, num_filters=128)
        self.encoder3 = encoder_block(block=conv_block, in_channels=128, num_filters=256)
        self.encoder4 = encoder_block(block=conv_block, in_channels=256, num_filters=512)
        self.bridge = conv_block(in_channels=512, num_filters=1024)
        self.dencoder1 = decoder_block(block=conv_block, num_filters=512)
        self.dencoder2 = decoder_block(block=conv_block, num_filters=256)
        self.dencoder3 = decoder_block(block=conv_block, num_filters=128)
        self.dencoder4 = decoder_block(block=conv_block, num_filters=64)
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same')
        self.sigmoid = nn.Sigmoid()

        self.attention1 = SpatialAttention(in_channels=64)
        self.attention2 = SpatialAttention(in_channels=128)
        self.attention3 = SpatialAttention(in_channels=256)
        self.attention4 = SpatialAttention(in_channels=512)

    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)
        b1 = self.bridge(p4)

        s4 = self.attention4(s4)
        s3 = self.attention3(s3)
        s2 = self.attention2(s2)
        s1 = self.attention1(s1)

        d1 = self.dencoder1(b1, s4)
        d2 = self.dencoder2(d1, s3)
        d3 = self.dencoder3(d2, s2)
        d4 = self.dencoder4(d3, s1)
        output = self.conv(d4)
        return self.sigmoid(output)

if __name__ == '__main__':
    x = torch.rand(2,1,512,512)
    model = UnetWithSA()
    y = model(x)
    print(y.shape)

