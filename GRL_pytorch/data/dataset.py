import os
import torch
import numpy as np
import pandas as pd
import scipy.io
from torch_geometric.data import InMemoryDataset, Data
import hashlib
import torch

import os
import torch
import numpy as np
import pandas as pd
import scipy.io
from torch_geometric.data import InMemoryDataset, Data
import hashlib
from graspologic.embed import AdjacencySpectralEmbed as ASE 
import networkx as nx
from node2vec import Node2Vec
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
import embeddings.spectral_embedding_methods as sem
from torch_geometric.utils import dense_to_sparse


class Dataset(InMemoryDataset):
    def __init__(self, root, dataset_name, x_type = 'one_hot', emb_size = 87, matrix_file='scs_desikan.mat', aligned_eigvecs = False, negs=False, num_nodes=87, transform=None, pre_transform=None, fc_paths=None, perturbar=False, labels_path = '/datos/projects/ssilva/GNNBAA/data/original_data/HCP_behavioral.csv'):
        self.matrix_file = matrix_file
        self.x_type = x_type
        self.dataset_name = dataset_name
        self.negs = negs
        self.num_nodes = num_nodes
        self.emb_size = emb_size
        self.fc_paths = fc_paths
        self.reference_matrix_sc = 'lol'
        self.reference_matrix_fc = 'lol'
        self.aligned_eigvecs = aligned_eigvecs
        self.perturbar = perturbar
        self.labels_path = labels_path
        
        self.unique_id = self._generate_unique_id(dataset_name, matrix_file, negs, num_nodes, x_type, emb_size, aligned_eigvecs)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    
    @property
    def raw_file_names(self):
        return [self.matrix_file]

    @property
    def processed_file_names(self):
        return [f'data_{self.unique_id}.pt']

    def _generate_unique_id(self, dataset_name, matrix_file, negs, num_nodes, x_type, emb_size, aligned_eigvecs):
        params_str = f"{dataset_name}-{matrix_file}-{negs}-{num_nodes}-{x_type}-{emb_size}-{aligned_eigvecs}"
        return hashlib.md5(params_str.encode()).hexdigest()

    def download(self):
        pass  # Implement if necessary

    def process(self):
        label_data = self._load_label_data()
        mat = scipy.io.loadmat(self.raw_paths[0])
        matrix = self._get_matrix(mat)
        subject_list = mat['subject_list'].squeeze()
        data_list = [self._process_subject(subject_name, matrix[:, :, idx], label_data, idx) for idx, subject_name in enumerate(subject_list) if subject_name in label_data]
        self.data, self.slices = self.collate([d for d in data_list if d is not None])
        torch.save((self.data, self.slices), self.processed_paths[0])


    def _load_label_data(self):
        label_data = pd.read_csv(self.labels_path)
        
        if self.dataset_name == 'Gender':
            return dict(zip(label_data['Subject'], label_data['Gender']))
        
        elif self.dataset_name == 'Age':
            return dict(zip(label_data['Subject'], label_data['Age']))


    def _get_matrix(self, mat):
        chosen_matrix_name = 'scs' if self.matrix_file.endswith('scs_desikan.mat') else 'fcs'
        return mat.get(chosen_matrix_name, None)


    def _process_subject(self, subject_name, subject_matrix, label_data, idx):
        
        print(f'procesing subject {subject_name}')
        try:
            # sc 
            sc = self._get_sc(subject_matrix)
            edge_index_sc, edge_weight_sc = self._convert_to_edge_index(sc)
            
            #fc
            fc = self._get_fc(subject_name)
            edge_index_fc, edge_weight_fc = self._convert_to_edge_index(fc)

            # features
            x = self._get_x('SC', sc, subject_name, idx)
            x_fc = self._get_x('FC', fc, subject_name, idx)

            # labels
            y = self._get_label(subject_name, label_data)
            
            # eigenvalues and eigenvectors
            if self.aligned_eigvecs:
                EigVecs, EigVals = self._get_aligned_eigvecs_and_vals(sc, 'SC', subject_name)
                EigVecs_fc, EigVals_fc = self._get_aligned_eigvecs_and_vals(fc, 'FC', subject_name)
            else:
                EigVecs, EigVals = self._get_eigvecs_and_vals(sc)
                EigVecs_fc, EigVals_fc = self._get_eigvecs_and_vals(fc)
                
            if y is not None:
                return Data(x=x, x_fc = x_fc, fc = fc, sc = sc, edge_index_fc = edge_index_fc, edge_weight_fc = edge_weight_fc, edge_index_sc = edge_index_sc, edge_weight_sc = edge_weight_sc, y = y, EigVals = EigVals, EigVecs = EigVecs, EigVecs_fc = EigVecs_fc, EigVals_fc = EigVals_fc)
        
        except Exception as e:
            print(f"Error processing subject {subject_name}: {e}")
        return None


    def _get_fc(self, subject_name):
        
        path_to_subject_fc = os.path.join(f'{self.fc_paths}/{subject_name}')
        fc_list = [np.load(os.path.join(path_to_subject_fc, fc)) for fc in os.listdir(path_to_subject_fc) if ('corr' in fc)]

        fc_list_processed = []
        for fc in fc_list:
            fc = fc[:, :].astype(np.float32)
            
            if self.num_nodes == 68:
                fc = fc[19:, 19:].astype(np.float32)
                
            if self.num_nodes == 167:
                fc = fc[:, :].astype(np.float32)
  
            if self.negs == False:
                fc = np.where(fc < 0, 0, fc)
                
            fc_list_processed.append(fc)
        
        if fc_list_processed != []:
            fc_new = np.array(fc_list_processed)
          
        return torch.tensor(fc_new, dtype=torch.float32).view(-1, self.num_nodes)
    
    
    def _get_sc(self, M):
        
        if self.num_nodes==87:
            M = M[:, :].astype(np.float32)
        else:
            M = M[19:, 19:].astype(np.float32)
            
        M = M + M.T - np.diag(M.diagonal())
                
        # if self.perturbar:
        #     nodo1, nodo2 = self.perturbar
        #     # Intercambia las filas
        #     M[[nodo1, nodo2], :] = M[[nodo2, nodo1], :]
        #     # Intercambia las columnas
        #     M[:, [nodo1, nodo2]] = M[:, [nodo2, nodo1]]
        
        # if self.perturbar:
            # bloque1, bloque2 = self.perturbar
            # # Intercambia los bloques de filas
            # filas_bloque1 = list(range(bloque1, bloque1 + 4))
            # filas_bloque2 = list(range(bloque2, bloque2 + 4))
            # M[filas_bloque1 + filas_bloque2, :] = M[filas_bloque2 + filas_bloque1, :]
            # # Intercambia los bloques de columnas
            # M[:, filas_bloque1 + filas_bloque2] = M[:, filas_bloque2 + filas_bloque1]
            # Genera una permutación aleatoria de los índices
            # permutacion = np.random.permutation(M.shape[0])
            # Aplica la permutación aleatoria a las filas y columnas
            # M = M[permutacion, :][:, permutacion]
            
        return torch.tensor(M, dtype=torch.float32)
    
    def _convert_to_edge_index(self, M):

        edge_index = M.nonzero(as_tuple=False)
        edge_index = edge_index.t().contiguous()
        edge_weights = M[edge_index[0], edge_index[1]].view(-1, 1)
                
        return edge_index, edge_weights
    
    def _get_aligned_eigvecs_and_vals(self, matrix, sc_or_fc, subject_name):
        matrix_np = matrix.numpy() 
        L = laplacian(matrix_np, normed=True)
        eigenvalues, eigenvectors = eigh(L)
        
        if subject_name == 100206:
            if sc_or_fc == 'SC':
                self.reference_matrix_sc = eigenvectors
            elif sc_or_fc == 'FC':
                self.reference_matrix_fc = eigenvectors
        else:
            if sc_or_fc == 'SC':
                eigenvectors = self._align_Xs(eigenvectors, self.reference_matrix_sc)
            elif sc_or_fc == 'FC':
                eigenvectors = self._align_Xs(eigenvectors, self.reference_matrix_fc)
                
        return torch.from_numpy(eigenvectors[:, :]), torch.from_numpy(eigenvalues[:])
    
    
    def _get_eigvecs_and_vals(self, matrix):
        
        matrix = np.where(matrix < 0, 0, matrix)

        L = laplacian(matrix, normed=True)
        eigenvalues, eigenvectors = eigh(L)
                
        return torch.from_numpy(eigenvectors[:, :]), torch.from_numpy(eigenvalues[:])


    def _get_label(self, subject_name, label_data):
        label = label_data[subject_name]
        
        if self.dataset_name == 'Gender':
            t =  torch.tensor([0 if label == 'M' else 1], dtype=torch.int64)
            
        elif self.dataset_name == 'Age':
            age_mapping = {'22-25': 0, '26-30': 1, '31-35': 2}
            t =  torch.tensor([age_mapping.get(label, 3)], dtype=torch.int64) if label in age_mapping else None
        
        else: 
            raise ValueError("Unsupported dataset_name")

        return t.to(torch.float32)
    
    def _get_x(self, sc_or_fc, matrix, subject_name, idx):
    
        if self.x_type == 'SPE' or self.x_type == 'Transformer':
            return torch.eye(self.num_nodes, dtype=torch.float32)
        
        if self.x_type == 'one_hot':
            return torch.eye(self.num_nodes, dtype=torch.float32)
        
        if self.x_type == 'aligned_veps':
            matrix = matrix.numpy() 
            L = laplacian(matrix, normed=True)
            _, eigenvectors = eigh(L)
            veps = eigenvectors[:, : self.emb_size]
            
            if sc_or_fc == 'SC':
                if subject_name == 100206:
                    self.reference_matrix_sc = veps
                else:
                    veps = self._align_Xs(veps, self.reference_matrix_sc)
                return torch.from_numpy(veps)
            
            else:
                if subject_name == 100206:
                    self.reference_matrix_fc = veps
                else:
                    veps = self._align_Xs(veps, self.reference_matrix_fc)
                return torch.from_numpy(veps)
        
        if self.x_type == 'veps_in_original_space':
            
            matrix = matrix.numpy() 
            L = laplacian(matrix, normed=True)
            vaps, eigenvectors = eigh(L)
            veps = eigenvectors[:, : self.emb_size]
            
            vaps = vaps[:self.emb_size]
            L_reconstructed = np.dot(np.dot(veps, np.diag(vaps)), veps.T)
            
            return torch.from_numpy(L_reconstructed)
        
        if self.x_type == 'veps':
            
            matrix = matrix.numpy() 
            L = laplacian(matrix, normed=True)
            _, eigenvectors = eigh(L)
            veps = eigenvectors[:, : self.emb_size]
        
            return torch.from_numpy(veps)
        
        if self.x_type == 'aligned_rdgp':
            
            ase = ASE(n_components = self.emb_size)
            X = ase.fit_transform(matrix.numpy())
            
            if sc_or_fc == 'SC':
                if subject_name == 100206:
                    self.reference_matrix_sc = X
                else:
                    X = self._align_Xs(X, self.reference_matrix_sc)
                return torch.from_numpy(X)
            
            else:
                if subject_name == 100206:
                    self.reference_matrix_fc = X
                else:
                    X = self._align_Xs(X, self.reference_matrix_fc)
                return torch.from_numpy(X)
            
        
        if self.x_type == 'rdgp':
            ase = ASE(n_components = self.emb_size)
            X = torch.tensor(ase.fit_transform(matrix.numpy()))
            return X

        if self.x_type == 'bondi':
            file = f'/home/personal/Documents/2023_2/tesis/GNN-BAA/GRL_pytorch/data/embedings/bondi_{sc_or_fc}_{self.emb_size}.npy'
            X_total = np.load(file)
            X = X_total[idx, :, :]
            return torch.from_numpy(X).to(torch.float32)
        
        if self.x_type == 'mase':
            file = f'/home/personal/Documents/2023_2/tesis/GNN-BAA/GRL_pytorch/embeddings/offline_embedings/mase_{sc_or_fc}_{self.emb_size}.npy'
            X = np.load(file)
            return torch.from_numpy(X).to(torch.float32)
        
        if self.x_type == 'eigen_vector_centrality':
            vaps, veps = torch.linalg.eig(matrix)

            vaps_abs = torch.abs(vaps.real)
            indice_max = torch.argmax(vaps_abs)
            veps_dominante= veps[:, indice_max].real
            centralidad_valor_propio = veps_dominante / torch.norm(veps_dominante, 1)

            return torch.diag(centralidad_valor_propio)
        
        if self.x_type == 'fc':
            return self._get_fc(subject_name)
        
        if self.x_type == 'sc':
            return matrix
        
        raise ValueError("Unsupported x_type")
    
    
    def _align_Xs(self, veps, reference_matrix):
        """
        An auxiliary function that Procrustes-aligns two embeddings.
        Parameters
        ----------
        X1 : an array-like with the embeddings to be aligned
        X2 : an array-like with the embeddings to align X1 to
        Returns
        -------
        X1_aligned : the aligned version of X1 to X2.
        """

        V,_,Wt = np.linalg.svd(veps.T @ reference_matrix) # voy hallando la transformacion para cada uno de ellos
        U = V @ Wt
        aligned_veps = veps @ U

        return aligned_veps

