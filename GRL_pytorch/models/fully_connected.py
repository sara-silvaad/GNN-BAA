import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class FullyConnected(nn.Module, ABC):
    def __init__(self, input_size, num_classes):
        super(FullyConnected, self).__init__()
        self.model_name ='FullyConnected'
        self.input_size = input_size
        # self.input_size = int((input_size - 87)/2)
        self.fc1 = nn.Linear(self.input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)

    @abstractmethod
    def forward(self, data):
        pass

class FullyConnectedFC(FullyConnected):
    def __init__(self, num_nodes, num_classes):
        super(FullyConnectedFC, self).__init__(num_nodes**2, num_classes)

    def forward(self, data):
        # x = data.fc
        # xx = x.reshape(-1, x.shape[1], x.shape[1])
        
        # weird_list_alt = [self.extract_upper_triangle(t) for t in xx]
        # xx = torch.stack(weird_list_alt)
                
        # x = xx.view(-1, self.input_size)
        
        device = torch.device("cuda")
        fc = data.fc.to(device)
        
        x = fc.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x, '_', '_'
    
    def extract_upper_triangle(self, t):
        
        indices = torch.triu_indices(row=t.size(0), col=t.size(1), offset=1)
        
        return t[indices[0], indices[1]]

class FullyConnectedSC(FullyConnected):
    def __init__(self, num_nodes, num_classes):
        super(FullyConnectedSC, self).__init__(num_nodes**2, num_classes)

    def forward(self, data):
        device = torch.device("cuda")
        sc = data.sc.to(device)
        x = sc.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, '_', '_'

class FullyConnectedSCFC(FullyConnected):
    def __init__(self, num_nodes, num_classes):
        # Duplicamos el input_size porque vamos a concatenar dos vectores de num_nodes**2 cada uno
        super(FullyConnectedSCFC, self).__init__(2 * num_nodes**2, num_classes)

    def forward(self, data):
        device = torch.device("cuda")
        fc = data.fc.to(device)
        sc = data.sc.to(device)
        
        x_fc = fc.view(-1, self.input_size//2)
        x_sc = sc.view(-1, self.input_size//2)
        
        x = torch.cat((x_fc, x_sc), dim=-1)  # Concatenamos a lo largo de la última dimensión
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, '_', '_'

