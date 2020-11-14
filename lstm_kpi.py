from numpy.core.defchararray import add
import torch
from torch.nn import LSTM
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn.modules import dropout
from torch.optim import Adam
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence


def load_data(path) -> np.ndarray:

    df = pd.read_csv(path)

    # test_df = pd.read_csv(path+"/KPI_test.csv")
    return df[['value', 'label', 'kpi_id']]


class Predictor(nn.Module):
    def __init__(self, input_dim=1, hid_dim=100, num_layer=1, dropout_p=0.2, output_dim=2):
        super(Predictor, self).__init__()
        self.lstm = LSTM(input_dim, hid_dim, num_layer, dropout=dropout_p)
        self.linear = nn.Linear(hid_dim, output_dim,
                                bias=True)
        self.loss_func = nn.CrossEntropyLoss()
        self.hid_dim = hid_dim
        self.input_dim = input_dim

    def forward(self, x):
        h_0 = torch.zeros(1, x.shape[1], self.hid_dim).cuda()
        c_0 = torch.zeros(1, x.shape[1], self.hid_dim).cuda()
        out, (h, c) = self.lstm(x, (h_0, c_0))
        preds = self.linear(out)
        return preds


if __name__ == "__main__":
    train_csv_path = "./data/KPI_train.csv"
    train_df = load_data(train_csv_path)
    groups = train_df.groupby('kpi_id')
    kpi_ids = pd.unique(train_df['kpi_id'])
    num_epoch = 100
    batch_size = 48
    seq_length = 16

    for kpi_id, kpi_df in groups:
        model = Predictor()
        model.cuda()
        train_size = int(len(kpi_df) * 0.8)
        optimizer = Adam(model.parameters(), lr=0.001)
        values = kpi_df['value'].values
        labels = kpi_df['label'].values
        train_x = values[:train_size]
        train_y = labels[:train_size]
        train_x = torch.FloatTensor(train_x).cuda()
        train_y = torch.LongTensor(train_y).cuda()
        batch_x = list()
        batch_y = list()
        for i in range(batch_size):
            j = train_size - i
            batch_x.append(train_x[j:])
            batch_y.append(train_y[j:])

        batch_x = pad_sequence(batch_x)
        batch_y = pad_sequence(batch_y)
        batch_x = batch_x.unsqueeze(-1)
        batch_y = batch_y.unsqueeze(-1)
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        loss_value = 0
        for i in range(num_epoch):
            preds = model(batch_x)
            loss = model.loss_func(preds.view(-1, 2), batch_y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            # if i % 100 == 0:
        print("{} loss: {}".format(kpi_id, loss_value))
        test_x = values[train_size:]
        test_y = labels[train_size:]
        test_x = torch.FloatTensor(test_x).view(-1, 1, 1).cuda()
        test_y = torch.LongTensor(test_y).view(-1).cuda()

        model.eval()
        logits = model(test_x)
        logits=logits.detach_().view(-1,2)
        logits = nn.functional.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        correct = torch.eq(preds, test_y).sum().item()
        acc = np.array(correct) / len(test_x)
        # f1 = f1_score(preds, test_y)
        print("{}, acc: {}".format(kpi_id, acc))
