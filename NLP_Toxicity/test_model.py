import pandas as pd
from torch.utils.data import DataLoader
from dataset import BinaryDataset, TrainData, structure_dataset
from model import BinaryModel
import torch
from preprocessing import text_preprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### Test_df Import #####
test_df = pd.read_csv('./data/test.csv').iloc[:100]

##### Text Preprocessing #####
test_df = text_preprocessing(test_df)

##### Dataset Structure #####
test_dataset = TrainData(test_df, max_seq_len=150)
test_dataset = structure_dataset(test_dataset)
x_test, y_test = test_dataset[0], test_dataset[1]
bin_test_dataset = BinaryDataset(x_test, y_test)
test_dataloader = DataLoader(bin_test_dataset, shuffle=False, batch_size=1)

# prepare the trained model
model = BinaryModel()
model.load_state_dict(torch.load('outputs/multi_head_binary.pth'))
model.to(device)
model.eval()


correct_pred = 0
for i, test_sample in enumerate(test_dataloader):
    print(f"SAMPLE {i}")
    # extract the features and labels
    features = test_sample['features'].resize_(test_sample['features'].size()[0], 150*300).to(device)
    target1 = test_sample['label1'].to(device)
    target2 = test_sample['label2'].to(device)
    target3 = test_sample['label3'].to(device)
    target4 = test_sample['label4'].to(device)
    target5 = test_sample['label5'].to(device)
    target6 = test_sample['label6'].to(device)
    
    outputs = model(features)
            
    # get all the labels
    all_labels = []
    for out in outputs:
        if out >= 0.5:
            all_labels.append(1)
        else:
            all_labels.append(0)
    
    targets = (target1, target2, target3, target4, target5, target6)
    
    # get all the targets in int format from tensor format
    all_targets = []
    for target in targets:
        all_targets.append(int(target.squeeze(0).detach().cpu()))
    
    if all_labels == all_targets:
        correct_pred += 1
            
    print(f"PREDICTED LABELS: {all_labels}")
    print(f"TRUE LABELS: {all_targets}")

accuracy_score = correct_pred / x_test.shape[0]

print(f'The accuracy of this model is {round(accuracy_score*100)}%')