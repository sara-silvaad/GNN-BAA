import torch
import numpy as np
import random
import os
import scipy
import sys
from torch_geometric.utils import dense_to_sparse
sys.path.append('C:/Users/marti/OneDrive/Desktop/facultad/Tesis/GNN-BAA/data_processing/hcp-download-script')
#from utils import calculate_fc_matrix, concatenate_timeseries
NUM_NODES = 87

def flip_sign(x):
    signs = torch.randint(0, 2, (x.shape[0], 1))*2 - 1
    return x*signs

def substract_random_ones(x, max):
    x = x.view(-1, NUM_NODES, NUM_NODES)
    matrices = torch.zeros(x.shape)
    for i in range(x.shape[0]):
        num_neg_ones = torch.randint(0, max-1, (1,)).item()  

        if num_neg_ones > 0:
            # Select unique random positions along the diagonal
            indices = torch.randperm(NUM_NODES)[:num_neg_ones]

            # Set these positions to -1
            matrices[i, indices, indices] = -1
    return (x+matrices).view(-1, NUM_NODES)

def add_random_ones(x):
    x = x.view(-1, NUM_NODES, NUM_NODES)
    matrices = torch.zeros(x.shape)
    for i in range(x.shape[0]):
        # Number of 1's to add
        num_ones = torch.randint(0, 51, (1,)).item()  # Random number between 0 and 50

        # Generate random positions for the 1's
        if num_ones > 0:
            rows = torch.randint(0, NUM_NODES, (num_ones,))
            cols = torch.randint(0, NUM_NODES, (num_ones,))
            matrices[i, rows, cols] = 1
        
    return (x + matrices).view(-1,NUM_NODES)
    
def diagonal_veps_matrices(x):

    def skewed_random_number(max_value=86, scale=10):
        # si se sube scale va hay mas chances de usar vaps altos
        random_exponential = np.random.exponential(scale)
        scaled_number = np.floor(random_exponential / (max_value + scale) * max_value)
        final_number = min(scaled_number, max_value)
        return int(final_number)
    
    x = x.view(-1, NUM_NODES, NUM_NODES)
    matrices = torch.zeros(x.shape)
    for i in range(x.shape[0]):
        n = skewed_random_number(86, 10)
        matrices[i] = torch.diag(x[i,:,n])
        
    return matrices.view(-1,NUM_NODES)

def one_region_of_ones(x):
    def create_diagonal_matrix(size, ranges):
        """Crea una matriz diagonal de tamaño 'size', con 1s en los rangos especificados y 0s en los demás lugares."""
        diag = np.zeros(size)  # Crea un vector diagonal inicializado a cero
        for start, end in ranges:
            diag[start:end] = 1  # Establece 1s en los rangos especificados
        matrix = np.diag(diag)  # Crea la matriz diagonal
        return matrix

    def elegir_matriz():

        # Definir los rangos para cada matriz
        ranges_dict = {
            '1': [(0, 19)],
            '2': [(19, 53)],
            '3': [(53, 87)],
            '4': [(0, 53)],
            '5': [(0, 19), (53, 87)],
            '6': [(19, 87)],
            '7': [(0,87)]
        }

        # Crear las matrices diagonales
        matrices = {key: create_diagonal_matrix(NUM_NODES, ranges) for key, ranges in ranges_dict.items()}

        # Sortear un número del 1 al 7 para decidir qué matriz usar
        key_selected = str(random.randint(1, 7))
        selected_matrix = matrices[key_selected]
        return selected_matrix

    x = x.view(-1, NUM_NODES, NUM_NODES)
    matrices = torch.zeros(x.shape)
    for i in range(x.shape[0]):
        matrices[i] = torch.tensor(elegir_matriz())
    
    return matrices.view(-1,NUM_NODES)

def sliding_window_da(path_to_data, path_to_save_augmented_data, num_windows = 10):
    for subject in os.listdir(path_to_data):
        path_to_ts = os.path.join(path_to_data, subject)
        
        try:
            ts_list = [scipy.io.loadmat(os.path.join(path_to_ts, ts))['dtseries'] for ts in os.listdir(path_to_ts)]
            if ts_list:
                ts_len = ts_list[0].shape[1]
            
                rate = ts_len // num_windows
                
                borders = np.arange(0, ts_len + rate, rate)  # Extend range to ensure 1065 is included
                borders[-1] = ts_len

                for i, border in enumerate(borders[1:]):
                    ts_list_per_window = [ts_list[j][:, borders[i]:border] for j in range(len(ts_list))] # genera una list con todas las ts pero solo en el rango de una ventana
                    concatenated = concatenate_timeseries(ts_list_per_window)
                    save_subject = os.path.join(path_to_save_augmented_data, subject)
                    os.makedirs(save_subject, exist_ok=True)
                    try:
                        _ = calculate_fc_matrix(concatenated, 'correlation', i, save_subject) # saves as path / subject / fc_i.npy
                    except ValueError:
                        print(f'Check subject {subject}')
            else:
                print(f'Subject {subject} doesnt have timeseries')
        except OSError:
            print(f'Subject {subject} is corrupted')



# def add_connections_for_node(edge_index_sc, edge_weight_sc, target_node, num_nodes, new_weight):
#     # Generar conexiones para el nodo objetivo con todos los otros nodos
#     target_edges_from = torch.full((num_nodes,), target_node, dtype=torch.long)
#     target_edges_to = torch.arange(num_nodes)
#     new_edges = torch.stack([target_edges_from, target_edges_to], dim=0)

#     # Combinar nuevas conexiones con las existentes
#     updated_edge_index_sc = torch.cat([edge_index_sc, new_edges], dim=1)
    
#     # Crear pesos para las nuevas conexiones
#     new_weights = torch.full((num_nodes,), new_weight)
    
#     # Combinar nuevos pesos con los existentes
#     updated_edge_weight_sc = torch.cat([edge_weight_sc, new_weights.unsqueeze(1)], dim=0)
    
#     return updated_edge_index_sc, updated_edge_weight_sc

# def transform_to_super_node(edge_index_sc, edge_weight_sc, target_node, num_nodes, new_weight):
#     total_nodes = edge_index_sc.max().item() + 1
#     num_graphs = total_nodes // num_nodes
    
#     # Inicializar listas para guardar los resultados
#     updated_edge_indices = []
#     updated_edge_weights = []
    
#     # Procesar cada grafo en el batch
#     for i in range(num_graphs):
#         start_idx = i * num_nodes
#         end_idx = (i + 1) * num_nodes
        
#         # Extraer los índices de las conexiones que pertenecen al grafo actual
#         graph_mask = (edge_index_sc[0] >= start_idx) & (edge_index_sc[0] < end_idx)
#         graph_edge_index = edge_index_sc[:, graph_mask] - start_idx
#         graph_edge_weight = edge_weight_sc[graph_mask]
        
#         # Transformar el nodo objetivo en supernodo
#         updated_edge_index, updated_edge_weight = add_connections_for_node(
#             graph_edge_index, graph_edge_weight, target_node, num_nodes, new_weight
#         )
        
#         # Ajustar los índices del grafo transformado para mantener el formato batch
#         updated_edge_index += start_idx
        
#         # Guardar los resultados
#         updated_edge_indices.append(updated_edge_index)
#         updated_edge_weights.append(updated_edge_weight)
    
#     # Combinar los grafos transformados de vuelta en el formato batch original
#     final_edge_index = torch.cat(updated_edge_indices, dim=1)
#     final_edge_weight = torch.cat(updated_edge_weights, dim=0)
    
#     return final_edge_index, final_edge_weight


def add_spurious_connections(A, intensidad, num_nodos=87):
    """
    Agrega una intensidad específica a todas las conexiones entre pares de nodos en cada grafo del batch,
    excluyendo la diagonal para evitar modificar las auto-conexiones.
    Luego, convierte las matrices densas actualizadas a representaciones esparsas.

    Parámetros:
    - A (torch.Tensor): Tensor de tamaño [batch * num_nodos, num_nodos] representando las matrices de adyacencia.
    - intensidad (float): Intensidad a agregar a las conexiones.
    - num_nodos (int): Número de nodos por grafo (default: 87).
    - batch_size (int): Tamaño del batch (default: 64).

    Retorna:
    - edge_index_final (torch.LongTensor): Índices de las aristas actualizadas.
    - edge_weight_final (torch.Tensor): Pesos de las aristas actualizadas.
    """
    batch_size = A.size(0)//87

    # Reshape de A a [batch, num_nodos, num_nodos]
    A_reshaped = A.view(batch_size, num_nodos, num_nodos)

    # Crear una máscara para excluir la diagonal
    mascara_excluir_diagonal = 1 - torch.eye(num_nodos, dtype=A.dtype, device=A.device)
    mascara_excluir_diagonal = mascara_excluir_diagonal.unsqueeze(0).repeat(batch_size, 1, 1)

    # Agregar intensidad a todas las parejas de nodos, excluyendo la diagonal
    A_reshaped += intensidad * mascara_excluir_diagonal

    # Inicializar listas para los resultados
    edge_indices = []
    edge_weights = []

    for k in range(batch_size):
        # Extraer la matriz de adyacencia para el grafo k
        A_grafo = A_reshaped[k]

        # Convertir la matriz densa a esparsa
        edge_index, edge_weight = dense_to_sparse(A_grafo)

        # Ajustar los índices para el batch
        edge_index = edge_index + (k * num_nodos)

        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)

    # Concatenar todos los grafos en el batch
    edge_index_final = torch.cat(edge_indices, dim=1)
    edge_weight_final = torch.cat(edge_weights, dim=0)

    return edge_index_final, edge_weight_final

def add_super_node(A, nodo_objetivo, intensidad, num_nodos=87):
    """
    Agrega intensidad a la fila y columna del nodo objetivo en cada grafo del batch,
    excluyendo la diagonal para evitar sumar dos veces la intensidad en el nodo objetivo.
    Luego, convierte las matrices densas actualizadas a representaciones esparsas.

    Parámetros:
    - A (torch.Tensor): Tensor de tamaño [batch * num_nodos, num_nodos] representando las matrices de adyacencia.
    - nodo_objetivo (int): Índice del nodo objetivo (0 a 86).
    - intensidad (float): Intensidad a agregar a las conexiones del nodo objetivo.
    - num_nodos (int): Número de nodos por grafo (default: 87).
    - batch_size (int): Tamaño del batch (default: 64).

    Retorna:
    - edge_index_final (torch.LongTensor): Índices de las aristas actualizadas.
    - edge_weight_final (torch.Tensor): Pesos de las aristas actualizadas.
    """

    # Reshape de A a [batch, num_nodos, num_nodos]
    batch_size = A.size(0)//87
    A_reshaped = A.view(batch_size, num_nodos, num_nodos)

    # Crear una máscara para la diagonal
    mascara_diagonal = torch.eye(num_nodos, dtype=A.dtype, device=A.device).unsqueeze(0).repeat(batch_size, 1, 1)

    for k in range(batch_size):
        # Agregar intensidad a la fila del nodo objetivo, excluyendo la diagonal
        A_reshaped[k, nodo_objetivo, :] += intensidad
        A_reshaped[k, nodo_objetivo, nodo_objetivo] -= intensidad  # Excluir la diagonal

        # Agregar intensidad a la columna del nodo objetivo, excluyendo la diagonal
        A_reshaped[k, :, nodo_objetivo] += intensidad
        A_reshaped[k, nodo_objetivo, nodo_objetivo] -= intensidad  # Excluir la diagonal

    # Inicializar listas para los resultados
    edge_indices = []
    edge_weights = []

    for k in range(batch_size):
        # Extraer la matriz de adyacencia para el grafo k
        A_grafo = A_reshaped[k]

        # Convertir la matriz densa a esparsa
        edge_index, edge_weight = dense_to_sparse(A_grafo)

        # Ajustar los índices para el batch
        edge_index = edge_index + (k * num_nodos)

        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)

    # Concatenar todos los grafos en el batch
    edge_index_final = torch.cat(edge_indices, dim=1)
    edge_weight_final = torch.cat(edge_weights, dim=0)

    return edge_index_final, edge_weight_final
  
def main():
    sliding_window_da(path_to_data, path_to_save_augmented_data)
    
if __name__ == '__main__':
    path_to_data = 'C:/Users/marti/OneDrive/Desktop/facultad/Tesis/GNN-BAA/data_processing/data/87_nodes_downloaded_on_imerl/time_series'
    path_to_save_augmented_data = 'C:/Users/marti/OneDrive/Desktop/facultad/Tesis/GNN-BAA/data_processing/data/87_nodes_downloaded_on_imerl/augmented_data_10'
    main()