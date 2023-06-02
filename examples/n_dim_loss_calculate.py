import sys
sys.path.append('../')

# libraries
import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import argparse
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm

matplotlib.rcParams['figure.figsize'] = [18, 12]

# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics

parser = argparse.ArgumentParser()
parser.add_argument('--dim', default=3, help='dimension for loss values calculation')
parser.add_argument('--steps', default=20, help='steps for loss values calculation')
args = parser.parse_args()

# training hyperparameters
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -2
BATCH_SIZE = 512
EPOCHS = 25
# contour plot resolution
STEPS = int(args.steps)
DIM = int(args.dim)

class MLPSmall(torch.nn.Module):
    """ Fully connected feed-forward neural network with one hidden layer. """
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, 32)
        self.linear_2 = torch.nn.Linear(32, y_dim)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        return F.softmax(self.linear_2(h), dim=1)


class Flatten(object):
    """ Transforms a PIL image to a flat numpy array. """
    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()    
    

def train(model, optimizer, criterion, train_loader, epochs):
    """ Trains the given model with the given optimizer, loss function, etc. """
    model.train()
    # train model
    for _ in tqdm(range(epochs), 'Training'):
        for count, batch in enumerate(train_loader, 0):
            optimizer.zero_grad()
            x, y = batch

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    model.eval()

# download MNIST and setup data loaders
mnist_train = datasets.MNIST(root='../data', train=True, download=True, transform=Flatten())
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False)

# define model
model = MLPSmall(IN_DIM, OUT_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.CrossEntropyLoss()

# stores the initial point in parameter space
model_initial = copy.deepcopy(model)

train(model, optimizer, criterion, train_loader, EPOCHS)

model_final = copy.deepcopy(model)

# data that the evaluator will use when evaluating loss
x, y = iter(train_loader).__next__()
metric = loss_landscapes.metrics.Loss(criterion, x, y)

# calculate the n-dimensional loss values
loss_data_fin_n_dim = loss_landscapes.random_n_directions(model_final, metric, DIM, 20, STEPS, normalization='filter', deepcopy_model=True)

print(loss_data_fin_n_dim.shape)

# save the binary file
np.save('n_dim_loss_binary_files/mlp_cifar10_dim_' + str(DIM) + '_steps_' + str(STEPS) + '.npy', loss_data_fin_n_dim)
