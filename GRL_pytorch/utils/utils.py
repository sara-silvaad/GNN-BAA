import os
import shutil
import re
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def check_path(path_to_save, config_path='./config.py'):
        # Verificar si la ruta existe
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        exp_path = os.path.join(path_to_save, 'exp_001')
        os.makedirs(exp_path)
        shutil.copy(config_path, exp_path)
        return exp_path

    # Obtener una lista de las subcarpetas existentes que coinciden con el patrón 'exp_###'
    existing_exp = [d for d in os.listdir(path_to_save) if re.match(r'exp_\d{3}', d)]

    if not existing_exp:
        exp_path = os.path.join(path_to_save, 'exp_001')
        os.makedirs(exp_path)
        shutil.copy(config_path, exp_path)
        return exp_path

    # Encontrar el último número de experimento
    existing_exp.sort()
    last_exp = existing_exp[-1]
    last_exp_num = int(re.search(r'\d{3}', last_exp).group())
    
    # Asignar el siguiente número de experimento
    new_exp_num = last_exp_num + 1
    new_exp_str = f'exp_{new_exp_num:03d}'
    exp_path = os.path.join(path_to_save, new_exp_str)
    os.makedirs(exp_path)
    shutil.copy(config_path, exp_path)
    
    return exp_path


def data_aug_transformation(train_loader_orig, k, weight):
    augmented_data_list = []
    
    for data in train_loader_orig.dataset:
        # Número de nodos originales
        num_nodes = data.x.size(0)
        
        # Seleccionar un nodo existente para conectar con k nodos aleatorios
        node_to_connect = torch.randint(0, num_nodes, (1,)).item()  # Randomly select one node to add edges from
        
        # Seleccionar k nodos aleatoriamente para conectar con el nodo seleccionado
        selected_nodes = torch.randperm(num_nodes)[:k]
        
        # Evitar que el nodo seleccionado se conecte a sí mismo
        selected_nodes = selected_nodes[selected_nodes != node_to_connect]
        
        # Crear edge_index y edge_weight para el nodo seleccionado
        new_edges = torch.stack([torch.tensor([node_to_connect] * len(selected_nodes)), selected_nodes], dim=0)
        new_edge_weights = (torch.ones(len(selected_nodes)) * weight).unsqueeze(1)
        
        # Actualizar edge_index_sc y edge_weight_sc
        new_edge_index_sc = torch.cat([data.edge_index_sc, new_edges, new_edges.flip(0)], dim=1)
        new_edge_weight_sc = torch.cat([data.edge_weight_sc, new_edge_weights, new_edge_weights])
        
        # Crear el nuevo objeto Data con los bordes adicionales añadidos
        augmented_data = Data(x=data.x, 
                            edge_index_fc=data.edge_index_fc, 
                            edge_weight_fc=data.edge_weight_fc, 
                            edge_index_sc=new_edge_index_sc, 
                            edge_weight_sc=new_edge_weight_sc, 
                            y=data.y, 
                            EigVals=data.EigVals, 
                            EigVecs=data.EigVecs, 
                            EigVals_fc=data.EigVals_fc, 
                            EigVecs_fc=data.EigVecs_fc)
    
        augmented_data_list.append(augmented_data)
    
    # Crear un nuevo DataLoader con los datos aumentados
    return DataLoader(augmented_data_list, 
                        batch_size=train_loader_orig.batch_size, 
                        shuffle=True)
