import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F



class ResNet18(nn.Module):
    def __init__(self, n_classes):
        super(ResNet18, self).__init__()

        ''' 
        Input Size of the ResNet18 is (w, h, nc) = (224 x 224 x 3)
        calculating the convolutional layer output/feature map spatial dimention size is 
             conv = (w + 2(p) - f/s +1, h + 2(p) -f/s +1)
             w,h = width, height
             f = filter
             s = stride
        calculating the pool layer output size:
        pool = (w - f/s +1, h - f/s +1)
        '''


        # Residual Block 1
        # first conv layer input image size 224x224x3
        # filters = 64, kernel size = 7x7, stride = 2x2, padding = 3x3
        #  conv output = w + 2p -f/s +1 ==> 224 + 2(3) - 7/2 +1 ==> 112

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=(7,7),stride = (2,2), padding = (3,3))
        self.batchnorm1 = nn.BatchNorm2d(64)
        # maxpool output = 112 - 3/2 + 1 ==>  (56, 56, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size = (3,3), stride = (2,2), padding = (1,1))
        self.__init_params(self.conv1)
        # self.skip1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(2,2), padding=(0,0))


        #residual block 2 
        # second residual input image size 56 x 56 x 64
        # filters = 64, kernel size = 3x3, stride = 1x1, padding = 3x3
        self.conv2_1_1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv2_1_1)
        self.batchnorm2_1_1 = nn.BatchNorm2d(64)
        self.conv2_1_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv2_1_2)
        self.batchnorm2_1_2 = nn.BatchNorm2d(64)
        # # Skip Layer 2_1
        # self.skip2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(2,2), padding=(0,0))

        self.conv2_2_1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv2_2_1)
        self.batchnorm2_2_1 = nn.BatchNorm2d(64)
        self.conv2_2_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv2_2_2)
        self.batchnorm2_2_2 = nn.BatchNorm2d(64)
        # # Skip Layer 2_2
        # self.skip2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.droput2 = nn.Dropout(p = 0.2)

        


        #residual block 3
        self.conv3_1_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=(3,3),stride = (2,2), padding = (1,1))
        self.__init_params(self.conv3_1_1)
        self.batchnorm3_1_1 = nn.BatchNorm2d(128)
        self.conv3_1_2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv3_1_2)
        self.batchnorm3_1_2 = nn.BatchNorm2d(128)
        # Skip Layer block 3_1
        self.skip3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(2,2), padding=(0,0))

        self.conv3_2_1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv3_2_1)
        self.batchnorm3_2_1 = nn.BatchNorm2d(128)
        self.conv3_2_2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv3_2_2)
        self.batchnorm3_2_2 = nn.BatchNorm2d(128)
        # # Skip Layer block 3_2
        # self.skip3_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.droput3 = nn.Dropout(p=0.2)
        
        #residual block 4
        self.conv4_1_1 = nn.Conv2d(in_channels = 128, out_channels =256 , kernel_size=(3,3),stride = (2,2), padding = (1,1))
        self.__init_params(self.conv4_1_1)
        self.batchnorm4_1_1 = nn.BatchNorm2d(256)
        self.conv4_1_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv4_1_2)
        self.batchnorm4_1_2 = nn.BatchNorm2d(256)
        # Skip Layer  4_1
        self.skip4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(2,2), padding=(0,0))

        self.conv4_2_1 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv4_2_1)
        self.batchnorm4_2_1 = nn.BatchNorm2d(256)
        self.conv4_2_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv4_2_2)
        self.batchnorm4_2_2 = nn.BatchNorm2d(256)
        # # Skip Layer  4_2
        # self.skip4_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout4 = nn.Dropout(p=0.2)

        #Residual block 5 
        self.conv5_1_1 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=(3,3),stride = (2,2), padding = (1,1))
        self.__init_params(self.conv5_1_1)
        self.batchnorm5_1_1 = nn.BatchNorm2d(512)
        self.conv5_1_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv5_1_2)
        self.batchnorm5_1_2 = nn.BatchNorm2d(512)
        # Skip Layer 5_1
        self.skip5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=(2,2), padding=(0,0))

        self.conv5_2_1 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv5_2_1)
        self.batchnorm5_2_1 = nn.BatchNorm2d(512)
        self.conv5_2_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.__init_params(self.conv5_2_2)
        self.batchnorm5_2_2 = nn.BatchNorm2d(512)
        # Skip Layer 5_2
        # self.skip5_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout5 = nn.Dropout(p=0.2)

        # fully connected block
        self.avgpool = nn.AvgPool2d(kernel_size = (7,7),stride=(1,1))
        self.fc = nn.Linear(in_features=1*1*512, out_features = 1000)
        self.__init_params(self.fc)
        self.output = nn.Linear(in_features = 1000, out_features = n_classes)

    def forward(self, x):
        # residual block 1 connection
        x = F.relu(self.batchnorm1(self.conv1(x)))
        opt1 = self.maxpool1(x)

        # residual block 2 connection

        x = F.relu(self.batchnorm2_1_1(self.conv2_1_1(opt1)))
        x = self.batchnorm2_1_2(self.conv2_1_2(x))

        
        opt2_1 = F.relu(x+opt1)

        x = F.relu(self.batchnorm2_2_1(self.conv2_2_1(opt2_1)))
        x = self.batchnorm2_2_2(self.conv2_2_2(x))
        x = self.droput2(x)

        # opt2_2_2 = self.skip2_2(opt2_2_2)
        opt2_2 = F.relu(x + opt2_1)

        # Residual block 3 Connection

        x = F.relu(self.batchnorm3_1_1(self.conv3_1_1(opt2_2)))
        x = self.batchnorm3_1_2(self.conv3_1_2(x))

        opt2_2 = self.skip3_1(opt2_2)
        opt3_1 = F.relu(x + opt2_2)

        x = F.relu(self.batchnorm3_2_1(self.conv3_2_1(opt3_1)))
        x = self.batchnorm3_2_2(self.conv3_2_2(x))
        x = self.droput3(x)

        opt3_2 = F.relu(x + opt3_1)

        # Residual Block 4 Connection

        x = F.relu(self.batchnorm4_1_1(self.conv4_1_1(opt3_2)))
        x = self.batchnorm4_1_2(self.conv4_1_2(x))

        opt3_2 = self.skip4_1(opt3_2)
        opt4_1 = F.relu(x + opt3_2)

        x = F.relu(self.batchnorm4_2_1(self.conv4_2_1(opt4_1)))
        x = self.batchnorm4_2_2(self.conv4_2_2(x))
        x = self.dropout4(x)

        opt4_2 = F.relu(x + opt4_1)

        # Residual Block 5 Connection

        x = F.relu(self.batchnorm5_1_1(self.conv5_1_1(opt4_2)))
        x = self.batchnorm5_1_2(self.conv5_1_2(x))

        opt4_2 = self.skip5_1(opt4_2)
        opt5_1 = F.relu(x + opt4_2)

        x = F.relu(self.batchnorm5_2_1(self.conv5_2_1(opt5_1)))
        x = self.batchnorm5_2_2(self.conv5_2_2(x))
        x = self.dropout5(x)

        opt5_2 = F.relu(x + opt5_1)

        # Connecting FC Layer
        x = self.avgpool(opt5_2)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc(x))
        x = F.softmax(x, dim = 1)
        x = self.output(x)
        return x
    
    def __init_params(self, layer):
        init.kaiming_normal(layer.weight)                     





