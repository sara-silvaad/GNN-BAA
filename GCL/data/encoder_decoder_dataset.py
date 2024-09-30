import os
import torch
import numpy as np
import pandas as pd
import scipy.io
from torch_geometric.data import InMemoryDataset, Data
import hashlib
from config import LABELS_PATH
class EncoderDecoderDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, threshold, normalize, x_type, transform=None, pre_transform=None,  matrix_file='scs_desikan.mat'):
        
        self.matrix_file = matrix_file  # Path to the SC or FC matrix file
        self.x_type = x_type
        self.dataset_name = dataset_name
        self.unique_id = self.generate_unique_id(dataset_name, matrix_file)
        self.threshold = threshold
        self.normalize = normalize
        
        super(EncoderDecoderDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.matrix_file]

    def generate_unique_id(self, dataset_name, matrix_file):
        # Create a string representation of the parameters
        params_str = f"{dataset_name}-{matrix_file}"
        # Use a simple hash function to generate a unique ID
        return hashlib.md5(params_str.encode()).hexdigest()

    @property
    def processed_file_names(self):
        return [f'data_{self.unique_id}.pt']
    
    def download(self):
        # Implement this method to download your raw files if necessary
        pass

    def process(self):
        
        label_data = pd.read_csv(LABELS_PATH)
        gender_map = dict(zip(label_data['Subject'], label_data['Gender']))
        age_map = dict(zip(label_data['Subject'], label_data['Age']))  # Fixed to map ages correctly
        
        alt_name = self.matrix_file.split('/')[-1] 
        self.chosen_matrix_name = 'scs' if alt_name == 'scs_desikan.mat' else 'fcs'
    
        # Proceed with processing using the chosen_matrix_name
        mat = scipy.io.loadmat(self.raw_paths[0])
        
        matrix = mat.get(self.chosen_matrix_name, None)
        
        if matrix is None:
            raise ValueError(f"The specified matrix file {self.matrix_file} does not contain the expected matrix.")
        
        subject_list = mat['subject_list'].squeeze()

        data_list = []
        y_values = []  # Store y values to calculate num_classes
        for idx, subject_name in enumerate(subject_list):
            
            edge_index, edge_weight = self.convert_to_edge_index(matrix[:, :, idx])
            
            try:
                
                num_nodes = matrix[:, :, idx].shape[0]
                x = self.get_x(num_nodes)

                fc = self.get_fc(subject_name)
                
                if self.dataset_name == 'Gender':
                        
                    gender = gender_map.get(subject_name, None)
                    if gender == 'M':
                        y = torch.tensor([0], dtype=torch.float)
                    elif gender == 'F':
                        y = torch.tensor([1], dtype=torch.float)
                    else:
                        raise ValueError(f'Gender not found for subject {subject_name}')
                        
                elif self.dataset_name == 'Age':
                    
                    age = age_map.get(subject_name, None)
                    if age == '22-25':
                        y = torch.tensor([0], dtype=torch.float)
                    elif age == '26-30':
                        y = torch.tensor([1], dtype=torch.float)
                    elif age == '31-35':
                        y = torch.tensor([2], dtype=torch.float)
                    elif age == '36+':
                        continue
                    else:
                        raise ValueError(f'Age not found for subject {subject_name}')

                y_values.append(y.item())
                data_list.append(Data(x = x, edge_index = edge_index,  y = y, edge_weight = edge_weight, fc = fc))
                
            except Exception as e:
                print(e)
        
        # Use the `collate` method to create the dataset
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
          
    def get_x(self, num_nodes):
        
        if self.x_type == 'one_hot':
            x = torch.eye(num_nodes, dtype=torch.float)
            
        return x
    
    
    def get_fc(self, subject_name):
        
        fc_init = np.load(os.path.join(self.root, f'fc_{subject_name}.npy'))
        fc = np.where(fc_init < 0 , 0 , fc_init)
        fc = torch.tensor(fc, dtype= torch.float)
        
        return fc
    
    def convert_to_edge_index(self, adj_matrix):
        adj_matrix = adj_matrix.astype(np.float64)
        
        # # Convert adjacency matrix to edge_index tensor
        # edge_index = torch.tensor(adj_matrix.nonzero(), dtype=torch.int)

        full_adj_matrix = adj_matrix + adj_matrix.T - np.diag(adj_matrix.diagonal())

        if self.normalize == 'mean-variance':
            # Sumamos la diagonal de 1 y normalizamos con media y varianza
            matriz_2 = np.eye(87)
            full_adj_matrix = full_adj_matrix + matriz_2
            # Calcular la media y la varianza por columna
            media = full_adj_matrix.mean(axis=0)
            varianza = full_adj_matrix.var(axis=0)
            # Normalizar cada columna
            full_adj_matrix = (full_adj_matrix - media) / np.sqrt(varianza)

        if self.normalize == 'log':
            full_adj_matrix = np.log(full_adj_matrix + 1)

        if self.normalize == 'min-max':
            full_adj_matrix = (full_adj_matrix- full_adj_matrix.min())/ (full_adj_matrix.max()- full_adj_matrix.min())
            
        A = torch.tensor(full_adj_matrix, dtype=torch.float)

        n = A.shape[0]
        # Calculamos el índice del umbral
        i_thr = int(n*n/2 + (1-self.threshold)*(n*n/2))
        # Calculamos el valor de umbral. Nota: necesitamos convertir A a NumPy temporalmente para usar np.sort, luego convertirlo de nuevo a tensor
        thr = torch.sort(A.flatten())[0][i_thr]

        # Ponemos la diagonal inferior en 0
        A = A.triu(diagonal=1)

        # Aplicamos el umbral
        A[A < thr] = 0
        A[A >= thr] = 1

        # Convertimos a edge_index
        edge_index = A.nonzero(as_tuple=False).t().contiguous()

        # Dado que ya tenemos A como un tensor binario, multiplicamos directamente por el tensor original
        full_adj_matrix_tensor = torch.tensor(full_adj_matrix, dtype=torch.float)  # Convertimos full_adj_matrix a tensor si aún no lo hemos hecho
        filtered_adj_matrix = full_adj_matrix_tensor * A

        # Extraemos los pesos de los enlaces de la matriz de adyacencia filtrada
        edge_weights = filtered_adj_matrix[edge_index[0], edge_index[1]].view(-1, 1)

        return edge_index, edge_weights

    def len(self):
        return len(self.data.y)

    def calculate_num_classes(self):
        # Implement if needed, based on the dataset's labels
        pass

    def get_num_features(self):
        if hasattr(self, 'data') and 'x' in self.data:
            return self.data['x'].size(1)
        else:
            return 0
        

        

