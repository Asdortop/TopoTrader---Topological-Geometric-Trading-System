import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Truncate the notification (padding) from the right
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # Causal Convolution 1
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.SELU() 
        self.dropout1 = nn.Dropout(dropout)
        
        # Causal Convolution 2
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.SELU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
                                 
        # Residual connection matching
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.SELU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # return self.relu(out + res) # PDF doesn't explicitly show relu after add, but standard ResNet does. 
        # Checking PDF snippet: "return out + res" -> No ReLU after sum. 
        # OK, I will follow PDF snippet exactly for the forward.
        return out + res

class MarketTCN(nn.Module):
    def __init__(self, num_inputs=12, num_channels=[32, 32, 32, 32], kernel_size=3, dropout=0.2):
        super(MarketTCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding = (k-1) * d to ensure causality with Chomp
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
                                     
        self.network = nn.Sequential(*layers)
        
        # Final prediction head
        # We take the last channel output (32) to 1.
        self.linear = nn.Linear(num_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x input shape: (Batch, Channels, Length)
        y = self.network(x)
        
        # Take the last time step prediction
        # y is (Batch, Channels, Length). We want y[:, :, -1] -> (Batch, Channels)
        return self.sigmoid(self.linear(y[:, :, -1]))
