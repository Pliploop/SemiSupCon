import torch
import torch.nn as nn

####
# Convolution block like also used in the SampleCNN model from https://github.com/Spijkervet/CLMR
# This block is used in every variant of the TUNe architecture
####

def initialize(m):
    if isinstance(m, (nn.Conv1d)):

        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")


class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, stride, padding):
        super(ConvBlock, self).__init__()

        self.layers = []
        self.layers.append(nn.Conv1d(in_chan, out_chan, kernel_size=kernel, stride=stride, padding=padding))
        self.layers.append(nn.BatchNorm1d(out_chan))
        self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)

        self.layers.apply(initialize)

    def forward(self, x):

        out = self.layers(x)

        return out

class TunePlus(nn.Module):
    def __init__(self, in_channels=11, n_classes=1, eval_layer=None):
        super(TunePlus, self).__init__()

        padding = 0
        stride = 3
        self.pool = nn.MaxPool1d(3, stride=3)
        self.eval_layer = eval_layer
        
        self.conv1 = ConvBlock(1            , int(in_channels),  kernel=3, stride=1, padding=1)
        self.conv2 = ConvBlock(int(in_channels), in_channels*2,  kernel=3, stride=1, padding=1)
        self.conv3 = ConvBlock(in_channels*2,    in_channels*4,  kernel=3, stride=1, padding=1)
        self.conv4 = ConvBlock(in_channels*4,    in_channels*8,  kernel=3, stride=1, padding=1)
        self.conv5 = ConvBlock(in_channels*8,    in_channels*16, kernel=3, stride=1, padding=1)
        
        
        self.transposedConv6 = nn.ConvTranspose1d(in_channels*16, in_channels*8, kernel_size=3, stride=stride, padding=padding)
        self.transposedConv7 = nn.ConvTranspose1d(in_channels*8,  in_channels*4, kernel_size=3, stride=stride, padding=padding)
        self.transposedConv8 = nn.ConvTranspose1d(in_channels*4,  in_channels*2, kernel_size=3, stride=stride, padding=padding)
        self.transposedConv9 = nn.ConvTranspose1d(in_channels*2,  in_channels*1, kernel_size=3, stride=stride, padding=padding)

        self.conv6 = ConvBlock(in_channels*16,     in_channels*8, kernel=3, stride=1, padding=1)
        self.conv7 = ConvBlock(in_channels*8,      in_channels*4, kernel=3, stride=1, padding=1)
        self.conv8 = ConvBlock(in_channels*4,      in_channels*2, kernel=3, stride=1, padding=1)
        self.conv9 = ConvBlock(int(in_channels*2), in_channels*1, kernel=3, stride=1, padding=1)


        # tail part
        self.convDown1 = ConvBlock(int(in_channels*3), in_channels*2,  kernel=3, stride=1, padding=1)
        self.convDown2 = ConvBlock(in_channels*6,      in_channels*4,  kernel=3, stride=1, padding=1)
        self.convDown3 = ConvBlock(in_channels*12,     in_channels*8,  kernel=3, stride=1, padding=1)
        self.convDown4 = ConvBlock(in_channels*24,     in_channels*16, kernel=3, stride=1, padding=1)

        # extra tail layers
        self.convDown5 = ConvBlock(in_channels*16, in_channels*32, kernel=3, stride=1, padding=1)
        self.convDown6 = ConvBlock(in_channels*32, in_channels*32, kernel=3, stride=1, padding=1)
        self.convDown7 = ConvBlock(in_channels*32, in_channels*32, kernel=3, stride=1, padding=1)
        self.convDown8 = ConvBlock(in_channels*32, in_channels*32, kernel=3, stride=1, padding=1)
        self.convDown9 = ConvBlock(in_channels*32, 512, kernel=3, stride=1, padding=1)


        if not eval_layer:
            self.output_avg = nn.AvgPool1d(3)
            self.fc = nn.Linear(512, n_classes)
        elif self.eval_layer == 1:
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*32, n_classes)
        elif self.eval_layer == 2:
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*32, n_classes)
        elif self.eval_layer == 3:
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*32, n_classes)
        elif self.eval_layer == 4:
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*32, n_classes)
        elif self.eval_layer == 5:
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*16, n_classes)
        elif self.eval_layer == 6:
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*8, n_classes)
        elif self.eval_layer == 7:
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*4, n_classes)
        elif self.eval_layer == 8:
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*2, n_classes)
        elif self.eval_layer == 9:
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*1, n_classes)

        elif self.eval_layer == 10:
            eval_layer = 8
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*2, n_classes)
        elif self.eval_layer == 11:
            eval_layer = 7
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*4, n_classes)
        elif self.eval_layer == 12:
            eval_layer = 6
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*8, n_classes)
        elif self.eval_layer == 13:
            eval_layer = 5
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*16, n_classes)
        elif self.eval_layer == 14:
            eval_layer = 6
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*8, n_classes)
        elif self.eval_layer == 15:
            eval_layer = 7
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*4, n_classes)
        elif self.eval_layer == 16:
            eval_layer = 8
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*2, n_classes)
        elif self.eval_layer == 17:
            eval_layer = 9
            self.output_avg = nn.AvgPool1d(3*3**eval_layer)
            self.fc = nn.Linear(in_channels*1, n_classes)
        
        nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, x):
        c1 = self.conv1(x)
        if self.eval_layer == 17:
            output = self.output_avg(c1)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        if self.eval_layer == 16:
            output = self.output_avg(c2)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        if self.eval_layer == 15:
            output = self.output_avg(c3)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        if self.eval_layer == 14:
            output = self.output_avg(c4)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)
        if self.eval_layer == 13:
            output = self.output_avg(c5)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)

        # expansive
        u6 = self.transposedConv6(c5)
        u6 = torch.cat((u6, c4), axis=1)
        c6 = self.conv6(u6)
        if self.eval_layer == 12:
            output = self.output_avg(c6)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)

        u7 = self.transposedConv7(c6)
        u7 = torch.cat((u7, c3), axis=1)
        c7 = self.conv7(u7)
        if self.eval_layer == 11:
            output = self.output_avg(c7)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)

        u8 = self.transposedConv8(c7)
        u8 = torch.cat((u8, c2), axis=1)
        c8 = self.conv8(u8)
        if self.eval_layer == 10:
            output = self.output_avg(c8)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)

        u9 = self.transposedConv9(c8)
        u9 = torch.cat((u9, c1[:,:,:]), axis=1) # sum to 10
        c9 = self.conv9(u9)
        if self.eval_layer == 9:
            output = self.output_avg(c9)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)

        p9 = self.pool(c9)
        
        # tail with pink links
        newthrough1 = torch.cat((p9, c8), axis=1) # sum to 10
        c10 = self.convDown1(newthrough1)
        if self.eval_layer == 8:
            output = self.output_avg(c10)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p10 = self.pool(c10)
        
        newthrough2 = torch.cat((p10, c7), axis=1) # sum to 10
        c11 = self.convDown2(newthrough2)
        if self.eval_layer == 7:
            output = self.output_avg(c11)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p11 = self.pool(c11)

        newthrough3 = torch.cat((p11, c6), axis=1) # sum to 10
        c12 = self.convDown3(newthrough3)
        if self.eval_layer == 6:
            output = self.output_avg(c12)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p12 = self.pool(c12)

        newthrough4 = torch.cat((p12, c5), axis=1) # sum to 10
        c13 = self.convDown4(newthrough4)
        if self.eval_layer == 5:
            output = self.output_avg(c13)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p13 = self.pool(c13)
        

        # extra tail layers
        c14 = self.convDown5(p13)
        if self.eval_layer == 4:
            output = self.output_avg(c14)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p14 = self.pool(c14)

        c15 = self.convDown6(p14)
        if self.eval_layer == 3:
            output = self.output_avg(c15)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p15 = self.pool(c15)

        c16 = self.convDown7(p15)
        if self.eval_layer == 2:
            output = self.output_avg(c16)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p16 = self.pool(c16)

        c17 = self.convDown8(p16)
        if self.eval_layer == 1:
            output = self.output_avg(c17)
            output = self.fc(output.permute(0,2,1))
            return output.view(output.shape[0],-1)
        p17 = self.pool(c17)

        c18 = self.convDown9(p17)

        output = self.output_avg(c18)
        # output = self.fc(output.permute(0,2,1))

        return output.view(output.shape[0],-1)