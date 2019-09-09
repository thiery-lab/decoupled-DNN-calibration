import torch.nn as nn
import torchvision.transforms as transforms
import math


class DiabRetinModel(nn.Module):
    """
    Model that was used as fifth place solution
    for diabetic retinopathy dataset. Used from
    https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/
    """

    def __init__(self, input_dim=(512, 512), num_channel=3, dropout_rate=0, num_classes=2):
        super(DiabRetinModel, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 32, kernel_size=7, stride=2,
                               padding=3, padding_mode='same', bias=True)
        self.leakyrelu = nn.LeakyReLU(0.5, inplace=True)

        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)

        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, stride=1,
                                padding=1, padding_mode='same', bias=True)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                padding=1, padding_mode='same', bias=True)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                padding=1, padding_mode='same', bias=True)
        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                padding=1, padding_mode='same', bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc1 = nn.Linear(12544, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(512, num_classes*2)


    def forward(self, x):

        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.leakyrelu(x)

        x = self.maxpool2d(x)

        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.leakyrelu(x)

        x = self.maxpool2d(x)

        x = self.conv4(x)
        x = self.leakyrelu(x)
        x = self.conv5(x)
        x = self.leakyrelu(x)

        x = self.maxpool2d(x)

        x = self.conv6(x)
        x = self.leakyrelu(x)
        x = self.conv7(x)
        x = self.leakyrelu(x)
        x = self.conv8(x)
        x = self.leakyrelu(x)
        x = self.conv9(x)
        x = self.leakyrelu(x)

        x = self.maxpool2d(x)

        x = self.conv10(x)
        x = self.leakyrelu(x)
        x = self.conv11(x)
        x = self.leakyrelu(x)
        x = self.conv12(x)
        x = self.leakyrelu(x)
        x = self.conv13(x)
        x = self.leakyrelu(x)

        x = self.maxpool2d(x)
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.maxpool1d(x.unsqueeze(1))
        x = x.squeeze()

        # combine both eyes
        x = x.view(batch_size // 2, -1)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.maxpool1d(x.unsqueeze(1))
        x = x.squeeze()

        x = self.dropout(x)
        x = self.fc3(x)
        # reshape to batch_size
        x = x.view(batch_size, -1)

        return x

