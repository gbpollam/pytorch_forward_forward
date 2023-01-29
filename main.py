import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy.random as random
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import TensorDataset, DataLoader

from utils.prepare_data import prepare_data

# Set seed for reproducibility
random.seed(42)


def MNIST_loaders(train_batch_size=30000, test_batch_size=5000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def WISDM_loaders(file_path, train_batch_size=30000, test_batch_size=5000):
    x_train, y_train_hot, x_test, y_test_hot = prepare_data(file_path, 20, 10, scaler_type='minmax')

    # Create a TensorDataset
    dataset_train = TensorDataset(x_train, y_train_hot)
    dataset_test = TensorDataset(x_test, y_test_hot)

    # Create a DataLoader
    train_loader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


def make_positive_data(x, y):
    # Use y to create the new tensor to append in front of x
    y_append = y.view(y.shape[0], 2, 3)
    x_ = torch.cat((x, y_append), dim=1)
    return x_


def make_negative_data(x, y):
    # Use y to create the new tensor to append in front of x, but this time with 1 on the wrong spot
    y_append = torch.zeros(size=(y.shape[0], 6))
    for i in range(y.shape[0]):
        skip_int = (y[i] == 1.).nonzero().item()
        allowed_ints = [j for j in range(6) if j != skip_int]
        replace_one = random.choice(allowed_ints)
        y_append[i][replace_one] = 1
    y_append = y_append.view(y_append.shape[0], 2, 3)
    # Move the tensor to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_append = y_append.to(device)
    x_ = torch.cat((x, y_append), dim=1)
    return x_


class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).cuda()]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        # Normalize the previous activation
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        print("x_direction shape: ", x_direction.shape)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
    
if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = WISDM_loaders('dataset/WISDM_ar_v1.1_raw.txt')

    net = Net([784, 500, 500])
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    x_pos = make_positive_data(x, y)
    x_neg = make_negative_data(x, y)
    
    # for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
    #     visualize_sample(data, name)
    
    net.train(x_pos, x_neg)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
