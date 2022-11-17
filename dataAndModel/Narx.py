import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def seed_torch(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MyDataset(Dataset):
    def __init__(self,
                 path,
                 input_seqlen=4,
                 output_seqlen=1,
                 fea_num=8,
                 train_percent=1,
                 isTrain=True):
        # data_df = pd.read_csv(
        #     path,  header=None)
        data_df = pd.read_excel(
            path, header=0)
        values = data_df.values
        self.data_num = len(values)
        self.input_seqlen = input_seqlen
        self.output_seqlen = output_seqlen
        self.all_seqlen = self.input_seqlen + self.output_seqlen
        self.train_index = int(self.data_num * train_percent)
        self.data_seq = []
        self.target_seq = []
        # self.scaler = MinMaxScaler(feature_range=(-0.9, 0.9))
        self.scaler = StandardScaler()
        self.scaler_x_extern = StandardScaler()
        self.scaler_x = StandardScaler()

        values[:,:-2] = self.scaler_x_extern.fit_transform(values[:,:-2])
        values[:,-2:-1] = self.scaler_x.fit_transform(values[:,-2:-1])
        values[:,-1:] = self.scaler.fit_transform(values[:,-1:])


        for i in range(self.data_num - self.all_seqlen):
            # ph = values[i+input_seqlen-1:i+input_seqlen, -1:] + \
            #     np.zeros((input_seqlen, 1))
            # all = np.concatenate((values[i:i + input_seqlen, :-1], ph), -1)
            all = values[i:i + input_seqlen, :]
            self.data_seq.append(list(all))
            self.target_seq.append(values[i+input_seqlen, -1])

        if isTrain is True:
            self.data_seq = self.data_seq[:self.train_index]
            self.target_seq = self.target_seq[:self.train_index]

        else:
            self.data_seq = self.data_seq[self.train_index:]
            self.target_seq = self.target_seq[self.train_index:]

        self.data_seq = np.array(self.data_seq).reshape(
            (len(self.data_seq), -1, fea_num))
        self.target_seq = np.array(self.target_seq).reshape(
            (len(self.target_seq), -1, 1))

        self.data_seq = torch.from_numpy(self.data_seq).type(torch.float32)
        self.target_seq = torch.from_numpy(self.target_seq).type(torch.float32)

    def __getitem__(self, index):
        return self.data_seq[index], self.target_seq[index]

    def __len__(self):
        return len(self.data_seq)

    def get(self):
        return self.data_seq, self.target_seq

    def get_scaler(self):
        return self.scaler


def prepare_dataset(batch_size):
    train = MyDataset(path,
                      input_seqlen=INPUT_SEQLEN,
                      output_seqlen=OUTPUT_SEQLEN)
    # test = MyDataset(path,
    #                  input_seqlen=INPUT_SEQLEN,
    #                  output_seqlen=OUTPUT_SEQLEN,
    #                  isTrain=False)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(-1, 32)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out


def train(epochs):
    loss_train = []
    loss_test = []
    for epoch in range(epochs):
        train_loss = 0.
        model.train()
        # lr_scheduler.step()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x)
            batch_y = batch_y.view(-1, 1, 1)
            out = out.view(-1, 1, 1)
            loss = mse(out, batch_y)
            optimizer.zero_grad()
            # if (i == 0):
            #     loss.backward(retain_graph=True)
            # else:
            #     loss.backward()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print('epoch : {}, Train Loss: {:.6f},lr : {:.6f}'.format(epoch + 1, train_loss / (i + 1),
                                                                  optimizer.state_dict()['param_groups'][0]['lr']))
        # print('lr:',optimizer.state_dict()['param_groups'][0]['lr'])
        loss_train.append(train_loss / (i + 1))
        # test(loss_test, epoch)
    # plot_loss(loss_train, loss_test)
    model.eval()
    torch.save(model, 'condition_1_model.pkl')

def eval_train():
    train = MyDataset(path,
                      input_seqlen=INPUT_SEQLEN,
                      output_seqlen=OUTPUT_SEQLEN,
                      isTrain=True)
    train_loader = DataLoader(train, batch_size=128, shuffle=False)
    model.eval()
    real = []
    predict = []
    with torch.no_grad():
        for j, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            out = model(batch_x)
            out = out.to('cpu')
            out = out.numpy().reshape(batch_x.size(0), 1)
            batch_y = batch_y.to('cpu')
            batch_y = batch_y.numpy().reshape(batch_x.size(0), 1)

            y_hat = scaler.inverse_transform(out)
            y_real = scaler.inverse_transform(batch_y)
            # y_hat = out
            # y_real = batch_y
            predict.append(y_hat)
            real.append(y_real)

    predict = np.concatenate(predict)
    real = np.concatenate(real)
    mse = mean_squared_error(real, predict)
    mae = mean_absolute_error(real, predict)
    r2 = r2_score(real, predict)
    mape = mean_absolute_percentage_error(real, predict)
    plt.figure()
    plt.plot(predict[:, -1], label="y_hat")
    plt.plot(real[:, -1], label="y", c="r")
    plt.legend()
    plt.xlabel("numbers of test")
    plt.ylabel("values")
    plt.title("real vs predict")
    plt.show()
    # print("mae", mae)
    # print("rmse:", sqrt(mse))
    # print("r2:", r2)
    # print("mape:", mape)


def data_to_excel(predict, real):
    predict = predict[:, :]
    real = real[:, :]
    np.savetxt('test_predict.txt',
               predict[:, 0])
    np.savetxt('test_real.txt', real[:, 0])


def plot_loss(loss_train, loss_test):
    plt.figure()
    plt.plot(loss_train, label="train loss", c='g')
    plt.plot(loss_test, label="test loss", c='b')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("train and test")
    plt.show()


if __name__ == '__main__':
    seed_torch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = 'Condition 1.xlsx'
    INPUT_SEQLEN = 4
    OUTPUT_SEQLEN = 1
    epochs = 200
    LR = 0.0001
    batch_size = 128
    model = Net().to(device)
   
    # model = torch.load('ANN_lizhuren.pkl').to(device)
    data = MyDataset(path=path, input_seqlen=INPUT_SEQLEN,
                     output_seqlen=OUTPUT_SEQLEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # lr_scheduler = StepLR(optimizer, step_size=21, gamma=0.8, last_epoch=-1)
    mse = torch.nn.MSELoss()
    train_loader = prepare_dataset(batch_size)
    scaler = data.get_scaler()
    # trainx, trainy = data.get()
    # print(trainx.size())
    # print(trainy.size())

    train(epochs)
    # eval()
    eval_train()
