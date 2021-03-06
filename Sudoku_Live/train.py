from data_set import CustomDataset
from torch import nn
from torch import optim
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


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


dataset = CustomDataset()

# train_size = int(0.9 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
epochs = 35
print_every = 50
total_step = len(train_loader)

    
# Train the model
    
for epoch in range(epochs):

    model.train()

    for i, (images, labels) in enumerate(train_loader):

    
        b_x = Variable(images)
        b_y = Variable(labels)
        output = model(b_x)[0]               
        loss = criterion(output, b_y)
    

        optimizer.zero_grad()
        
        loss.backward()
  
        optimizer.step()
        
        if (i+1) % 30 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

torch.save(model.state_dict(), "output/best_model.pt")


#     # Test the model
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         # images = images.float()
#         output, last_layer = model(images)
#         prediction = torch.max(output, 1)[1].data.squeeze()
#         accuracy = (prediction == labels).sum().item() / float(labels.size(0))
#         pass
# print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)