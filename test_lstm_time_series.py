import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

NUM_EPOCHS = 20
NUM_FORECAST = 10
INIT_DISCARD_PERC = 0.33
LEARNING_RATE = 1e-3
NUM_HIDDEN_UNITS = 16
BATCH_SIZE = 10
SEQ_LENGTH = 30


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(
            hn[0]
        ).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


def load_csv():
    df = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(value=0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(value=0)
    df["Weighted_Price"] = df["Weighted_Price"].fillna(value=0)
    df["Open"] = df["Open"].fillna(method="ffill")
    df["High"] = df["High"].fillna(method="ffill")
    df["Low"] = df["Low"].fillna(method="ffill")
    df["Close"] = df["Close"].fillna(method="ffill")
    # For minute-based evaluation on Close:
    # df = df.set_index("Timestamp")
    # df = df["Close"].to_frame()
    # For date averaging
    df["date"] = pd.to_datetime(df["Timestamp"], unit="s").dt.date
    df = df.groupby("date").mean()
    # df = grouped_df["Close"].mean()
    # df = df.to_frame()

    # Discarding first third of dataset -> Low activity in trades
    df = df.iloc[int(len(df) * INIT_DISCARD_PERC) :].copy()
    return df


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for x, y in data_loader:
        output = model(x)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for x, _ in data_loader:
            y_star = model(x)
            output = torch.cat((output, y_star), 0)

    return output


def main():
    df = load_csv()

    target_sensor = "Close"
    features = list(df.columns.difference([target_sensor]))

    target = f"{target_sensor}_lead{NUM_FORECAST}"

    df[target] = df[target_sensor].shift(-NUM_FORECAST)
    df = df.iloc[:-NUM_FORECAST]

    test_frac = 0.8

    df_train = df.iloc[: int(len(df) * test_frac)].copy()
    df_test = df.iloc[int(len(df) * test_frac) :].copy()

    target_mean = df_train[target].mean()
    target_stdev = df_train[target].std()

    for c in df_train.columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()

        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev

    train_dataset = SequenceDataset(
        df_train, target=target, features=features, sequence_length=SEQ_LENGTH
    )

    train_dataset = SequenceDataset(
        df_train, target=target, features=features, sequence_length=SEQ_LENGTH
    )
    test_dataset = SequenceDataset(
        df_test, target=target, features=features, sequence_length=SEQ_LENGTH
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=NUM_HIDDEN_UNITS)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()

    for ix_epoch in range(NUM_EPOCHS):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
        test_model(test_loader, model, loss_function)
        print()

    train_eval_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model).numpy()
    df_test[ystar_col] = predict(test_loader, model).numpy()

    df_out = pd.concat((df_train, df_test))[[target, ystar_col]]

    for c in df_out.columns:
        df_out[c] = df_out[c] * target_stdev + target_mean

    print(df_out)

    plt.plot(df_out.index, df_out[f"{target_sensor}_lead{NUM_FORECAST}"])
    plt.plot(df_out.index, df_out[f"{ystar_col}"])
    plt.show()


if __name__ == "__main__":
    main()
