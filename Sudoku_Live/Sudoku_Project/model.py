from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=32,            
                kernel_size=3
                ),                              
            nn.ReLU(),   
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(32, 64, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),
            nn.Dropout(p = 0.25)              
        )
        self.out1 = nn.Sequential(
            nn.Linear(10816, 64),
            nn.ReLU(),
            nn.Dropout(p = 0.5)
        )
        self.out2 = nn.Linear(64, 9)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        output = self.out1(x)
        output = self.out2(output)
        return output, x
