import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.backends
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import csv

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# TODO: read data  
data = pd.read_csv('data\ml2022spring-hw1\covid.train.csv')
label = data['tested_positive']
data = data.drop(['tested_positive'], axis=1)
data = data.to_numpy()
label = label.to_numpy()

test_data = pd.read_csv('data\ml2022spring-hw1\covid.test.csv')
test_label = test_data['tested_positive']
test_data = test_data.drop(['tested_positive'], axis=1)
test_data = test_data.to_numpy()
test_label = test_label.to_numpy()

# TODO: select feature
def select(data, test_data, selected_features = all):
    if selected_features == all:
        return data, test_data
    else:
        return data[:, selected_features], test_data[:, selected_features]
    

data, test_data = select(data, test_data, selected_features=list(range(30,115)))

# TODO: DataSet
class COVID19Dataset(Dataset):
    def __init__(self, data, label = None):
        # for training
        if label is not None:
            self.data = torch.FloatTensor(data)
            self.label = torch.FloatTensor(label)
        # for testing
        else:
            self.data = torch.FloatTensor(data)
            self.label = label

    def __getitem__(self, index):
        if self.label is not None:
            return self.data[index], self.label[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

# TODO: set random seed
def same_seed(seed):
    torch.backends.cudnn.deterministic = True #cudnn 确定性模式， 每次运行出现相同输出
    torch.backends.cudnn.benchmark = False #基准测试会选择最快算法，导致运算时间不确定性
    np.random.seed(seed) #np随机种子确定，处理数据确定
    torch.manual_seed(seed) #torch随机种子确定，dataset
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# TODO: split dataset
def train_valid_split(dataset, valid_ratio, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_set, valid_set = random_split(dataset, [1 - valid_ratio, valid_ratio], generator=generator)
    return train_set, valid_set

# TODO: DataSet and DataLoader
same_seed(42)
train_set, valid_set = train_valid_split(COVID19Dataset(data, label), 0.2, 42)
print(f'train data size {len(train_set) ,len(train_set[0][0])} \nvalid data size {len(valid_set), len(valid_set[0][0])} ')

batch_size = 256

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

test_set = COVID19Dataset(test_data)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print(f'test data size {len(test_set)} ')

# TODO: define model
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
    def forward(self, x):
        return self.layers(x).squeeze(1)




# TODO: define train
def train(model, train_loader, valid_loader, num_epoches=3000, lr = 1e-5):
    criterion = nn.MSELoss(reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter('runs/hw1')
    step = 0
    early_stop_count = 0
    best_loss = float('inf')
    for epoch in range(num_epoches):
        model.train()
        losses = []

        train_bar = tqdm(train_loader, position=0, leave=True) # 9/9 是一batch多少step
        for x, y in train_bar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            losses.append(loss.detach().item())

            train_bar.set_description(f'Epoch [{epoch+1}/{num_epoches}], Loss: {loss.detach().item()}')

        mean_train_loss = sum(losses)/len(losses)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()
        losses = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            
            losses.append(loss.item())

        mean_valid_loss = sum(losses)/len(losses)
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            if not os.path.isdir('model\hw1'):
                os.makedirs('model\hw1')
            torch.save(model.state_dict(), 'model\hw1\\best_model.pt')
            print(f'Epoch [{epoch+1}/{num_epoches}], Loss: {mean_train_loss}, Valid Loss: {mean_valid_loss}')
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        if early_stop_count >= 400:
            print(f'the model is not improving for {early_stop_count} epochs, early stop')
            return
        
# TODO: start training\
# model = Classifier(input_dim=data.shape[1]).to(device)
# train(model, train_loader, valid_loader) 

def save_pred(preds, file):
    with open(file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'value'])
        for i, pred in enumerate(preds):
            writer.writerow([i, pred])

model = Classifier(input_dim=data.shape[1]).to(device)
model.load_state_dict(torch.load('model\hw1\\best_model.pt'))
def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

preds = predict(test_loader, model, device)
save_pred(preds, 'hw1/submission.csv')
