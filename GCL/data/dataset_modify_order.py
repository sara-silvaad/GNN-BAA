import os
import torch
import numpy as np
import pandas as pd
import scipy.io
from torch_geometric.data import InMemoryDataset, Data
import hashlib
from config import LABELS_PATH
import torch
import random

import os
import torch
import numpy as np
import pandas as pd
import scipy.io
from torch_geometric.data import InMemoryDataset, Data
from graspologic.embed import AdjacencySpectralEmbed as ASE
import hashlib
from config import LABELS_PATH , MATRIX_CORR_MAX

COUNT = []

class Dataset(InMemoryDataset):
    def __init__(self, root, dataset_name, threshold, normalize, x_type, negs, which_fc, num_nodes, transform=None, pre_transform=None, matrix_file='scs_desikan.mat'):
        self.matrix_file = matrix_file
        self.x_type = x_type
        self.dataset_name = dataset_name
        self.negs = negs
        self.threshold = threshold
        self.normalize = normalize
        self.which_fc = which_fc
        self.num_nodes = num_nodes
        self.unique_id = self._generate_unique_id(dataset_name, matrix_file, negs, normalize, which_fc, num_nodes, x_type)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.matrix_file]

    @property
    def processed_file_names(self):
        return [f'data_{self.unique_id}.pt']

    def _generate_unique_id(self, dataset_name, matrix_file, negs, normalize, which_fc, num_nodes, x_type):
        params_str = f"{dataset_name}-{matrix_file}-{negs}-{normalize}-{which_fc}-{num_nodes}-{x_type}"
        return hashlib.md5(params_str.encode()).hexdigest()

    def download(self):
        pass  # Implement if necessary

    def process(self):
        label_data = self._load_label_data()
        mat = scipy.io.loadmat(self.raw_paths[0])
        matrix = self._get_matrix(mat)
        subject_list = mat['subject_list'].squeeze()
        data_list = [self._process_subject(subject_name, matrix[:, :, idx], label_data) for idx, subject_name in enumerate(subject_list) if subject_name in label_data]
        self.data, self.slices = self.collate([d for d in data_list if d is not None])
        torch.save((self.data, self.slices), self.processed_paths[0])

    def _load_label_data(self):
        label_data = pd.read_csv(LABELS_PATH)
        return dict(zip(label_data['Subject'], zip(label_data['Gender'], label_data['Age'])))

    def _get_matrix(self, mat):
        chosen_matrix_name = 'scs' if self.matrix_file.endswith('scs_desikan.mat') else 'fcs'
        return mat.get(chosen_matrix_name, None)

    def _process_subject(self, subject_name, subject_matrix, label_data):
        try:
            # sc 
            sc = self._get_sc(subject_matrix)
            edge_index_sc, edge_weight_sc = self._convert_to_edge_index(sc)
            
            # fc
            if self.which_fc == 'Max':
                fc = self._get_fc(subject_name)
            elif self.which_fc == 'ours':
                fc = self._get_fc_backup(subject_name)
            else:
                print('FC selected doesnt exist')
            edge_index_fc, edge_weight_fc = self._convert_to_edge_index(fc)

            # features
            x = self._get_x(sc, subject_name)

            # labels
            y = self._get_label(subject_name, label_data)

            if y is not None:
                return Data(x=x, fc = fc, sc = sc, edge_index_fc = edge_index_fc, edge_weight_fc = edge_weight_fc, edge_index_sc = edge_index_sc, edge_weight_sc = edge_weight_sc, y = y)
        except Exception as e:
            print(f"Error processing subject {subject_name}: {e}")
        return None

    def _get_x(self, sc, subject_name):
        if self.x_type == 'one_hot':
            one_hot = torch.eye(self.num_nodes, dtype=torch.float)
            #return self._modify_node_order(one_hot)
            return one_hot
        
        if self.x_type == 'sc':
            return sc
        
        if self.x_type == 'sc_no_diagonal':
            rows = torch.arange(87).view(-1, 1)
            cols = [torch.cat((torch.arange(i), torch.arange(i + 1, 87))) for i in range(87)]
            cols = torch.stack(cols)
            result = sc[rows, cols]
            return result
        
        if self.x_type == 'veps_vaps':
            D = torch.diag(sc.sum(1))
            L = D - sc
            vaps, veps = torch.linalg.eigh(L)

            #concatenar los vaps a los veps
            vaps_reshaped = vaps[:87].repeat(87,1)
            veps_vaps = torch.cat((veps[:,:87], vaps_reshaped), dim =1)
            return veps_vaps
        
        if self.x_type == 'veps':
            D = torch.diag(sc.sum(1))
            L = D - sc
            
            #D_inv_sqrt = torch.diag(1.0 / (torch.sqrt((D).diagonal())))
            #L_norm = torch.mm(torch.mm(D_inv_sqrt, L), D_inv_sqrt)
            
            vaps, veps = torch.linalg.eigh(L)
            return veps[:, :5] #concatenar vaps
        
        if self.x_type == 'eigen_vector_centrality':
            # Calcular los valores propios y los vectores propios de la matriz de adyacencia
            vaps, veps = torch.linalg.eig(sc)
            # Encontrar el vector propio asociado al valor propio más grande
            # Nota: torch.linalg.eig puede devolver valores complejos, por lo que se maneja esto aquí
            vaps_abs = torch.abs(vaps.real)
            indice_max = torch.argmax(vaps_abs)
            veps_dominante= veps[:, indice_max].real
            # Normalizar el vector propio dominante para obtener las centralidades de valor propio
            centralidad_valor_propio = veps_dominante / torch.norm(veps_dominante, 1)
            # Crear la matriz diagonal con las centralidades de valor propio
            return torch.diag(centralidad_valor_propio)
        
        if self.x_type == 'fc':
            return self._get_fc(subject_name)
        
        if self.x_type == 'rdpg':
            ase = ASE(n_elbows= 2)
            X = torch.tensor(ase.fit_transform(sc.numpy()))
            if X.shape[1] != 2:
                COUNT.append(X.shape[1])
                print('\n')
                print(COUNT)
                print(len(COUNT))
                print('\n')
            return X
        
        raise ValueError("Unsupported x_type")

    def _get_fc_backup(self, subject_name):
        
        fc_init = np.load(os.path.join(f'{self.root}/{subject_name}', f'fc_{subject_name}_corr.npy'))
        
        # Our matrices have the first 19 subcortical nodes
        
        fc = fc_init[-(self.num_nodes-87):, -(self.num_nodes-87):].astype(np.float64)
        
        if self.negs == False:
            fc = np.where(fc < 0, 0, fc)
            
        return torch.tensor(fc, dtype=torch.float)
    
    def _get_fc(self, subject_name):
        
        data = scipy.io.loadmat(MATRIX_CORR_MAX)
    
        # Extract the subject list and 'fcs' matrix
        subject_list = data['subject_list']
        fcs_matrix = data['fcs']
        
        # Initialize subject_index as None
        subject_index = None
        
        # Iterate over the subject_list to find the index of the provided subject_id
        for i, subj in enumerate(subject_list):
            if subj[0] == subject_name:
                subject_index = i
                break
        
        # Check if the subject ID was found
        if subject_index is None:
            print(f"Subject ID {subject_name} not found.")
            return None
        
        # Extract the 'fcs' matrix for the specific subject
        subject_fcs = fcs_matrix[:self.num_nodes, :self.num_nodes, subject_index]
        
        # negs==False means i don't keep negative values
        if self.negs == False:
            subject_fcs = np.where(subject_fcs < 0, 0, subject_fcs)
        
        fc = torch.tensor(subject_fcs, dtype=torch.float)
        return self._modify_node_order(fc)
    
    def _modify_node_order(self, M):
        # Dimensiones de la matriz original
        n = M.shape[0]
        
        # Índices originales y destinos
        from_indices = torch.tensor(list(range(87)))
        numbers = list(range(87))
        random.shuffle(numbers)   
        to_indices = torch.tensor(numbers)
        
        # Crear un tensor para el nuevo orden
        index_mapping = torch.arange(n)
        
        # Actualizar el mapeo con los nuevos índices
        index_mapping[from_indices] = to_indices
        index_mapping[to_indices] = from_indices
        
        # Reordenar las filas y columnas
        new_M = M[index_mapping][:, index_mapping]
        
        return new_M
    
    def _get_sc(self, M):
        M = M[-(self.num_nodes-87):, -(self.num_nodes-87):].astype(np.float64)
        # M = M[:self.num_nodes, :self.num_nodes].astype(np.float64)
        M = M + M.T - np.diag(M.diagonal())
        M = self._normalize(M, self.normalize)
        M = self._apply_threshold(M)
        M_ = self._modify_node_order(M)
        return M_
    
    def _convert_to_edge_index(self, M):
        M = M.triu(diagonal=1)
        edge_index = M.nonzero(as_tuple=False).t().contiguous()
        edge_weights = M[edge_index[0], edge_index[1]].view(-1, 1)
        return edge_index, edge_weights
    
    def _apply_threshold(self, M):
        A = torch.tensor(M, dtype=torch.float)
        n = A.shape[0]
        # Calculamos el índice del umbral
        i_thr = int(n*n/2 + (1-self.threshold)*(n*n/2))
        # Calculamos el valor de umbral. Nota: necesitamos convertir A a NumPy temporalmente para usar np.sort, luego convertirlo de nuevo a tensor
        thr = torch.sort(A.flatten())[0][i_thr]
        # Aplicamos el umbral
        A[A < thr] = 0
        A[A >= thr] = 1
        M = torch.tensor(M, dtype=torch.float)  # Convertimos M a tensor si aún no lo hemos hecho
        return M * A

    def _normalize(self, M, tipo):
        if tipo == 'mean_variance':
            # Sumamos la diagonal de 1 y normalizamos con media y varianza
            matriz_2 = np.eye(self.num_nodes)
            M = M + matriz_2
            # Calcular la media y la varianza por columna
            media = M.mean(axis=0)
            varianza = M.var(axis=0)
            # Normalizar cada columna
            M = (M - media) / np.sqrt(varianza)

        if tipo == 'log':
            M = np.log(M + 1)

        if tipo == 'min_max':
            M = (M- M.min())/ (M.max()- M.min())
        return torch.tensor(M, dtype=torch.float)

    def _get_label(self, subject_name, label_data):
        gender, age = label_data[subject_name]
        if self.dataset_name == 'Gender':
            t =  torch.tensor([0 if gender == 'M' else 1], dtype=torch.int64)
        elif self.dataset_name == 'Age':
            age_mapping = {'22-25': 0, '26-30': 1, '31-35': 2}
            t =  torch.tensor([age_mapping.get(age, 3)], dtype=torch.int64) if age in age_mapping else None
        else: 
            raise ValueError("Unsupported dataset_name")
        #return torch.nn.functional.one_hot(t, num_classes=NUM_CLASSES).float()
        return t.float()