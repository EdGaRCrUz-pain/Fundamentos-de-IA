import torch
import torch.nn as nn
import torch.nn.functional as F


x = torch.tensor([-1.0, 1.0, 2.0, 3.0])
output = torch.softmax(x, dim=0)
print(output)

sm = nn.Softmax(dim=0)
output = sm(x)
print(output)


output = torch.sigmoid(x)
print(output)

s = nn.Sigmoid()
output = s(x)
print(output)


output = torch.tanh(x)
print(output)

t = nn.Tanh()
output = t(x)
print(output)


output = torch.relu(x)
print(output)

relu = nn.ReLU()
output = relu(x)
print(output)


output = F.leaky_relu(x)
print(output)

lrelu = nn.LeakyReLU()
output = lrelu(x)
print(output)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out
