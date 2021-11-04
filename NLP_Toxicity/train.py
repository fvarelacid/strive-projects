###### For the train_multi_head_binary.py file ######

# from dataset import make_dataset, BinaryDataset
from torch.utils.data import DataLoader
# from loss_functions import binary_loss_fn as loss_fn
from model import BinaryModel
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')



train_dataset = BinaryDataset(x_train, y_train)
train_dataloader = DataLoader(bin_train_dataset, shuffle=True, batch_size=1024)

# initialize the model
model = BinaryModel()

# training function
def train(model, dataloader, optimizer, loss_fn, train_dataset, device):
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        features = data['features'].to(device)
        target1 = data['toxic'].to(device)
        target2 = data['severe_toxic'].to(device)
        target3 = data['obscene'].to(device)
        target4 = data['threat'].to(device)
        target5 = data['insult'].to(device)
        target6 = data['identity_hate'].to(device)
        
        # zero-out the optimizer gradients
        optimizer.zero_grad()
        
        outputs = model(features)
        targets = (target1, target2, target3, target4, target5, target6)
        loss = loss_fn(outputs, targets)
        train_running_loss += loss.item()
        
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    return train_loss

# learning parameters
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 50
model.to(device)


# start the training
train_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_dataloader, optimizer, binary_loss_fn, bin_train_dataset, device
    )
    train_loss.append(train_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
torch.save(model.state_dict(), 'outputs/multi_head_binary.pth')
