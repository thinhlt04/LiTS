import torch
import torch.nn as nn


class encoder_block(nn.Module):
    def __init__(self, block, in_channels, num_filters):
        super().__init__()
        self.conv_block = block(in_channels = in_channels, num_filters=num_filters)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        s = self.conv_block(X)
        p = self.pool(s)
        return s, p 

class decoder_block(nn.Module):
    def __init__(self, block, num_filters):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=num_filters*2, out_channels=num_filters, kernel_size=2, stride=2)
        self.conv_block = block(in_channels=num_filters*2,num_filters=num_filters)

    def forward(self, x, skip_features):
        x = self.deconv(x)
        x = torch.cat((x, skip_features), 1)
        x = self.conv_block(x)
        return x

    
class conv_block(nn.Module):
    def __init__(self, in_channels, num_filters):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels=num_filters, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv_block(x)
    
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention
