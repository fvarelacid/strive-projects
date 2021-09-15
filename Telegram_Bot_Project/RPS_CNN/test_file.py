import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from main import Net

from handgestures.transform_image import transform_single_image

save_path = 'RPS_net.pth'
# model = torch.load(save_path)
# # print(model.fc1.weight)
# new_m = Net()
# new_m.load_state_dict(model)

model = Net()
model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
model.eval()