import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from abc import ABC, abstractmethod
from data.data_augmentation import substract_random_ones, one_region_of_ones,  add_super_node, add_spurious_connections
import random

class GCNEncoderDecoderClassifier(torch.nn.Module, ABC):
    def __init__(self, hidden_dims, num_classes, pooling_type='ave', concatenate=True, model_name='BaseModel', negs=False, num_nodes = 87, residual = False):
        super(GCNEncoderDecoderClassifier, self).__init__()
        self.model_name = model_name
        self.num_nodes = num_nodes
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.concatenate = concatenate
        self.residual = residual
        self.negs = negs
        self.convs = torch.nn.ModuleList()
        self.mlp = torch.nn.ModuleList()
        self.activations = []
        self.gradients_activations = []
        self.gradients_weights = []

        for i in range(len(hidden_dims) - 1):
            if self.residual:
                in_channels = sum(hidden_dims[:i+1])  # Suma acumulativa de las dimensiones anteriores
            else:
                in_channels = hidden_dims[i]
            
            out_channels = hidden_dims[i + 1]
            conv = GCNConv(in_channels, out_channels)
            self.convs.append(conv)

        self.classifier = torch.nn.Linear(hidden_dims[-1] if not concatenate else sum(hidden_dims[1:]), num_classes)

    def encoder(self, x, edge_index, edge_weights):
        layer_output = []
        layer_input = []
        self.activations = []
        self.gradients_activations = []

        for i, conv in enumerate(self.convs):
            if self.residual and i!= 0:
                x = torch.cat((x, layer_input[i-1]), dim = 1)
            layer_input.append(x)
            x = F.relu(conv(x, edge_index, edge_weights))
            layer_output.append(x)

        if self.concatenate and len(self.convs) > 1:
            node_emb = torch.cat(layer_output, dim=-1)
        else:
            node_emb = layer_output[-1]
        return node_emb

    def graph_embedding(self, node_emb, batch):
        if self.pooling_type == 'ave':
            graph_emb = global_mean_pool(node_emb, batch)
        elif self.pooling_type == 'sum':
            graph_emb = torch.sum(node_emb, dim=1).unsqueeze(-1)
        else:  # max pooling
            graph_emb = global_max_pool(node_emb, batch)
        return graph_emb

    def decoder(self, emb):
        if self.concatenate:
            node_emb = emb.view(-1, self.num_nodes, sum(self.hidden_dims[1:]))
        else:
            node_emb = emb.view(-1, self.num_nodes, self.hidden_dims[-1])
        node_emb_transpose = node_emb.transpose(1, 2)

        if self.negs:
            reconstructed_adj = torch.tanh(torch.matmul(node_emb, node_emb_transpose))
        else:
            reconstructed_adj = torch.relu(torch.matmul(node_emb, node_emb_transpose))

        reconstructed_adj = reconstructed_adj.view(-1, self.num_nodes)
        return reconstructed_adj

    @abstractmethod 
    def forward(self, data):
        pass

class EncoderClassifierSC(GCNEncoderDecoderClassifier):
    def forward(self, data):
        x, edge_index, edge_weights, batch = data.x, data.edge_index_sc, data.edge_weight_sc, data.batch
        
        if torch.cuda.is_available():
            device = torch.device("cuda")  
            x, edge_index, edge_weights, batch = x.to(device), edge_index.to(device), edge_weights.to(device), batch.to(device)
        else:
            device = torch.device("cpu")
            
        node_emb = self.encoder(x, edge_index, edge_weights)
        graph_emb = self.graph_embedding(node_emb, batch)
        classif_logits = torch.squeeze(self.classifier(graph_emb)).float()
        return classif_logits, x, graph_emb

class EncoderClassifierSCSuperNode(GCNEncoderDecoderClassifier):
    def __init__(self, *args, intensity_super_node= 500, **kwargs):
        super(EncoderClassifierSCSuperNode, self).__init__(*args, **kwargs)
        self.intensity_super_node = intensity_super_node

    def forward(self, data):
        x, sc, edge_index, edge_weights, batch = data.x, data.sc, data.edge_index, data.edge_weights, data.batch
            
        if self.training:
            node = random.randint(0,86)
            edge_index, edge_weights = add_super_node(sc, node, self.intensity_super_node, 87)
            
        if torch.cuda.is_available():
            device = torch.device("cuda")  
            x, edge_index, edge_weights, batch = x.to(device), edge_index.to(device), edge_weights.to(device), batch.to(device)
        else:
            device = torch.device("cpu")

        node_emb = self.encoder(x, edge_index, edge_weights)
        graph_emb = self.graph_embedding(node_emb, batch)
        classif_logits = torch.squeeze(self.classifier(graph_emb)).float()
        return classif_logits, x, graph_emb
    
    
class EncoderClassifierSCSpurious(GCNEncoderDecoderClassifier):
    def __init__(self, *args, intensity= 10, **kwargs):
        super(EncoderClassifierSCSpurious, self).__init__(*args, **kwargs)
        self.intensity = intensity

    def forward(self, data):
        x, sc, batch = data.x, data.sc, data.batch
            
        edge_index, edge_weights = add_spurious_connections(sc, self.intensity, 87)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")  
            x, edge_index, edge_weights, batch = x.to(device), edge_index.to(device), edge_weights.to(device), batch.to(device)
        else:
            device = torch.device("cpu")

        node_emb = self.encoder(x, edge_index, edge_weights)
        graph_emb = self.graph_embedding(node_emb, batch)
        classif_logits = torch.squeeze(self.classifier(graph_emb)).float()
        return classif_logits, x, graph_emb
    
class EncoderClassifierSCMaskAttributes(GCNEncoderDecoderClassifier):
    def __init__(self, *args, N = 87, **kwargs):
        super(EncoderClassifierSCMaskAttributes, self).__init__(*args, **kwargs)
        self.N = N

    def forward(self, data):
        x, edge_index, edge_weights, batch = data.x, data.edge_index_sc, data.edge_weight_sc, data.batch
        
        if self.training:
            x = substract_random_ones(x, self.N)
            
        if torch.cuda.is_available():
            device = torch.device("cuda:1")  
            x, edge_index, edge_weights, batch = x.to(device), edge_index.to(device), edge_weights.to(device), batch.to(device)
        else:
            device = torch.device("cpu")

        node_emb = self.encoder(x, edge_index, edge_weights)
        graph_emb = self.graph_embedding(node_emb, batch)
        classif_logits = torch.squeeze(self.classifier(graph_emb)).float()
        return classif_logits, x, graph_emb
    
class EncoderClassifierSCZoneMaskAttributes(GCNEncoderDecoderClassifier):
    def forward(self, data):
        x, edge_index, edge_weights, batch = data.x, data.edge_index_sc, data.edge_weight_sc, data.batch
        
        if self.training:
            x = one_region_of_ones(x)
            
        if torch.cuda.is_available():
            device = torch.device("cuda")  
            x, edge_index, edge_weights, batch = x.to(device), edge_index.to(device), edge_weights.to(device), batch.to(device)
        else:
            device = torch.device("cpu")

        node_emb = self.encoder(x, edge_index, edge_weights)
        graph_emb = self.graph_embedding(node_emb, batch)
        classif_logits = torch.squeeze(self.classifier(graph_emb)).float()
        return classif_logits, x, graph_emb
