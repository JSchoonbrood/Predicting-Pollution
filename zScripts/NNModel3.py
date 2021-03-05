import numpy as np
import pandas as pd
import torch
from torch import nn
import os
import sys
from pathlib import Path

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_files, process=True):

        dataframes = []
        for fname in csv_files:
            data = pd.read_csv(root_dir + fname)
            dataframes.append(data)
        self.dataframe = pd.concat(dataframes)

        #self.transform = transforms.Compose([transforms.ToTensor()])

        self.root_dir = root_dir
        self.process = process

        if process:
            self.processed_data = self.processData()

    def __len__(self):
        if self.process:
            return len(self.processed_data)
        else:
            return len(self.dataframe)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = idx.tolist()

        if self.process:
            x = np.array(self.processed_data.iloc[index][:4])
            y = np.array(self.processed_data.iloc[index][-1])
            return torch.from_numpy(x), torch.from_numpy(y)
        else:
            return torch.from_numpy(self.dataframe.iloc[index])

    def processData(self):
        data_to_process = self.dataframe.copy()

        numeric_attr = ['Mean_Speed', 'Estimated_Travel_Time', 'Traffic_Level', 'Total_Neighbours', 'Length']
        for col in numeric_attr:
            data_to_process[col] = pd.to_numeric(data_to_process[col], errors='coerce')

        data_to_process['Rank'] = data_to_process['Rank'].astype("category")

        train_attr = data_to_process[numeric_attr]

        train_attr = pd.get_dummies(data_to_process, columns=['Rank'])

        return train_attr


class Model(torch.nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x


def run():
    if sys.platform == "linux" or sys.platform == "linux2":
        current_dir = os.path.dirname(__file__)
        data_dir = os.path.join(Path(current_dir).parent, 'RANKED_CSV/')
    elif sys.platform == "win32":
        current_dir = os.path.dirname(__file__)
        data_dir = os.path.join(Path(current_dir).parent, r'RANKED_CSV\\')

    training_files = os.listdir(data_dir)

    BATCH_SIZE = 64
    EPOCH = 20
    LEARNING_RATE = 1e-3

    #model = Model(embedding_size=0, num_numerical_cols=5, output_size=8, layers=[30, 30, 30])

    ds = Dataset(root_dir=data_dir, csv_files=training_files)
    loader = torch.utils.data.DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()


    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            b_x = torch.autograd.Variable(batch_x)
            b_y = torch.autograd.Variable(batch_y)

            input = b_x.float()
            target = b_y.float()

            prediction = model(input)     # input x and predict based on x

            loss = loss_fn(prediction, target)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

        threshold = 0.1
        errors = (prediction - target) ** 2  # Squared error
        acc = (errors < threshold).float().mean()
        error = errors.mean()
        #accuracy.append(torch.Tensor.detach(acc).numpy()[0])
        #loss_list.append(torch.Tensor.detach(error).numpy()[0])
        print (accuracy)
        print (loss_list)

if __name__ == '__main__':
    run()
