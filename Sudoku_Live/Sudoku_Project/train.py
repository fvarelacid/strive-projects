import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


from torchvision import datasets, transforms

# Define a transform to normalize the data (Preprocessing)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5)) ])

# Download and load the training data
trainset    = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Download and load the test data
testset    = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)


# Hyperparameters for our network
input_size   = 784
hidden_sizes = [128, 64]
output_size  = 10

# Build a feed-forward network
# model = nn.Sequential(OrderedDict([
#           ('fc1', nn.Linear(input_size, hidden_sizes[0])),
#           ('relu1', nn.ReLU()),
#           ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
#           ('relu2', nn.ReLU()),
#           ('logits', nn.Linear(hidden_sizes[1], output_size))]))

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.dropout1 = nn.Dropout(p = 0.25)
        self.dropout2 = nn.Dropout(p = 0.50)
        self.fc1 = nn.Sequential(
			nn.Linear(7*7*64, 64),
			nn.ReLU())
        self.fc2 = nn.Sequential(
			nn.Linear(64, 10),
			nn.ReLU())
            
    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.dropout1(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.softmax(out, dim = 1)
        return out


model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
epochs = 10
print_every = 40
acc_list = []
total_step = len(trainloader)

for e in range(epochs):
    running_loss = 0
    print(f"Epoch: {e+1}/{epochs}")

    model.train()

    for i, (images, labels) in enumerate(iter(trainloader)):

        # Flatten MNIST images into a 784 long vector
        # images.resize_(images.size()[0], 784)
        
        optimizer.zero_grad()
        
        output = model(images)   # 1) Forward pass
        loss = criterion(output, labels) # 2) Compute loss
        loss.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
        
        total = labels.size(0)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)
        
        if (i + 1) % 25 == 0:
	        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {} / {}'.format(e + 1, epochs, i + 1, total_step, loss.item(), correct, total))

torch.save(model.state_dict(), "output/best_model.pt")