import os
import torch
import numpy as np
import pandas as pd
import scipy.io
from torch_geometric.data import InMemoryDataset, Data
import hashlib
# from config import LABELS_PATH
import torch
from torch_geometric.utils import dense_to_sparse

import os
import torch
import numpy as np
import pandas as pd
import scipy.io
from torch_geometric.data import InMemoryDataset, Data
import hashlib
from graspologic.embed import AdjacencySpectralEmbed as ASE 

from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

from config import LABELS_PATH


class Dataset(InMemoryDataset):
    def __init__(self, root, dataset_name, x_type, emb_size = None, matrix_file='scs_desikan.mat', num_nodes=68, negs=False, threshold=1, transform=None, pre_transform=None, fc_paths='/datos/projects/ssilva/GNNBAA/data/data_desikan_task/corr_matrices/', all_nodes = True, signal_vals = 'SC', perturbar=False):
        self.matrix_file = matrix_file
        self.x_type = x_type
        self.dataset_name = dataset_name
        self.negs = negs
        self.num_nodes = num_nodes
        self.emb_size = emb_size
        self.fc_paths = fc_paths
        self.reference_matrix_fc = 'lol'
        self.all_nodes = all_nodes
        self.signal_vals = signal_vals
        self.pe = 'lol'
        self.perturbar = perturbar
        
        self.unique_id = self._generate_unique_id(dataset_name, matrix_file, negs, num_nodes, x_type, emb_size)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    
    @property
    def raw_file_names(self):
        return [self.matrix_file]

    @property
    def processed_file_names(self):
        return [f'data_{self.unique_id}.pt']

    def _generate_unique_id(self, dataset_name, matrix_file, negs, num_nodes, x_type, emb_size):
        params_str = f"{dataset_name}-{matrix_file}-{negs}-{num_nodes}-{x_type}-{emb_size}-float32"
        return hashlib.md5(params_str.encode()).hexdigest()

    def download(self):
        pass  # Implement if necessary

    def process(self):

        mat = scipy.io.loadmat(self.raw_paths[0])
        subject_list = mat['subject_list'].squeeze()

        data_list = []
        for idx, subject_name in enumerate(subject_list):
            subject_data_list = self._process_subject(subject_name, idx)
            if subject_data_list is not None:
                data_list.extend(subject_data_list)  # Extend the main list with the returned list

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


    def _process_subject(self, subject_name, idx):
        print(f'Processing subject {subject_name}')
        data_list = []

        try:
            # Get the FC matrices and labels for the subject
            fcs, labels = self._get_multiple_fc(subject_name)

            # Process each FC matrix separately
            for fc, label in zip(fcs, labels):
                # Convert FC to edge index and weights
                edge_index_fc, edge_weight_fc = self._convert_to_edge_index(fc)

                # Get the feature matrix 'x_fc'
                x_fc = self._get_x(fc, subject_name, idx)

                # Create a Data object for each FC matrix
                y = torch.tensor(label, dtype=torch.int32)  # Assuming y should be a tensor with the label
                
                data = Data(x=x_fc, fc=fc, edge_index=edge_index_fc, edge_weight=edge_weight_fc, y=y)

                # Append the Data object to the data_list
                data_list.append(data)

        except Exception as e:
            print(f"Error processing subject {subject_name}: {e}")

        # Return the list of Data objects
        return data_list

    
    def _get_multiple_fc(self, subject_name):
        
        # self.fc_paths es una variable nueva con el path a la carpeta que sea que quieras usar donde estan las fc, ponele donde hay una sola o varias aumentadas
        path_to_subject_fc = os.path.join(f'{self.fc_paths}{subject_name}')
        fc_list = [np.load(os.path.join(path_to_subject_fc, fc)) for fc in os.listdir(path_to_subject_fc) if ('corr' in fc)]

        fc_list_processed = []
        labels = []
        for fc, path_fc in zip(fc_list, os.listdir(path_to_subject_fc)):
            
            if 'LANGUAGE' in path_fc:
                label = 0
            elif 'MOTOR' in path_fc:
                label = 1
            # elif 'GAMBLING' in path_fc:
            #     label = 2
            # elif 'EMOTION' in path_fc:
            #     label = 3
            else:
                print('Not any of the right classes')
                continue
            
            labels.append(label)
            fc = fc[:, :].astype(np.float32)
            
            if self.num_nodes == 68:
                fc = fc[19:, 19:].astype(np.float32)
  
            if self.negs == False:
                fc = np.where(fc < 0, 0, fc)
                
            fc_list_processed.append(fc)
                
        fc = np.array(fc_list_processed)
                            
        return torch.tensor(fc, dtype=torch.float32), labels
  
    
    def _convert_to_edge_index(self, M):
        edge_index, edge_weights = dense_to_sparse(M)
        return edge_index, edge_weights
    
    
    
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
    
    
    def _get_aligned_eigvecs_and_vals(self, matrix, subject_name):
        matrix_np = matrix.numpy() 
        L = laplacian(matrix_np, normed=True)
        eigenvalues, eigenvectors = eigh(L)
        
        if subject_name == 100206:
            self.reference_matrix_fc = eigenvectors
        else:
            eigenvectors = self._align_Xs(eigenvectors, self.reference_matrix_fc)
                
        return torch.from_numpy(eigenvectors[:, :]), torch.from_numpy(eigenvalues[:])
    
    
    def _get_eigvecs_and_vals(self, matrix):
        matrix_np = matrix.numpy() 
        L = laplacian(matrix_np, normed=True)
        eigenvalues, eigenvectors = eigh(L)
                
        return torch.from_numpy(eigenvectors[:, :]), torch.from_numpy(eigenvalues[:])

    
    def _get_x(self, matrix, subject_name, idx):
        
        if self.x_type == 'SPE' or self.x_type == 'Transformer':
            return torch.eye(self.num_nodes, dtype=torch.float32)
        
        if self.x_type == 'transf_pos_enc':
            
            if subject_name == 100206:
                position = np.arange(self.num_nodes)[:, np.newaxis]
                div_term = np.exp(np.arange(0, self.emb_size) * -(np.log(10000.0) / self.emb_size))
                pe = np.zeros((self.num_nodes, self.emb_size))
                pe[:, 0::2] = np.sin(position * div_term[0::2])
                pe[:, 1::2] = np.cos(position * div_term[1::2])
                
                self.pe = pe

            return torch.from_numpy(self.pe).to(torch.float32)
                
        if self.x_type == 'random_noise':
            if subject_name == 100206:
                x = np.random.normal(0, 1, size=(self.num_nodes, self.emb_size))
                self.pe = x
                
            return torch.from_numpy(self.pe).to(torch.float32)

        if self.x_type == 'large_one_hot':
            
            large_one_hot = torch.eye(170, dtype=torch.float32)
            
            return large_one_hot[:, :self.num_nodes]
            
        
        if self.x_type == 'one_hot':
            return torch.eye(self.num_nodes, dtype=torch.float32)
        
        if self.x_type == 'aligned_veps':
            
            fc_np = matrix.numpy() 
            L = laplacian(fc_np, normed=True)
            _, eigenvectors = eigh(L)
            veps = eigenvectors[:, : self.emb_size]
            
            if subject_name == 100206:
                self.reference_matrix_fc = veps
            else:
                veps = self._align_Xs(veps, self.reference_matrix_fc)
            
            return torch.from_numpy(veps)
        
        if self.x_type == 'veps_in_original_space':

            sc_np = matrix.numpy() 
            L = laplacian(sc_np, normed=True)
            vaps, eigenvectors = eigh(L)
            veps = eigenvectors[:, : self.emb_size]
            
            vaps = vaps[:self.emb_size]
            L_reconstructed = np.dot(np.dot(veps, np.diag(vaps)), veps.T)
            
            return torch.from_numpy(L_reconstructed)
        
        if self.x_type == 'veps':
            
            sc_np = matrix.numpy() 
            L = laplacian(sc_np, normed=True)
            _, eigenvectors = eigh(L)
            veps = eigenvectors[:, : self.emb_size]
        
            return torch.from_numpy(veps)
        
        if self.x_type == 'aligned_rdgp':
            
            ase = ASE(n_components = self.emb_size)
            X = ase.fit_transform(matrix.numpy())
            
            if subject_name == 100206:
                self.reference_matrix = X
            else:
                X = self._align_Xs(X)
                
            return torch.from_numpy(X)
        
        if self.x_type == 'rdgp':
            ase = ASE(n_components = self.emb_size)
            X = torch.tensor(ase.fit_transform(matrix.numpy()))
            return X
        
        
        if self.x_type == 'bondi':
            file = f'/datos/projects/ssilva/GNNBAA/GRL_pytorch/embeddings/embedings/bondi_{self.emb_size}.npy'
            X_total = np.load(file)
            X = X_total[idx, :, :]
            return torch.from_numpy(X).to(torch.float32)
        
        if self.x_type == 'mase':
            file = f'/datos/projects/ssilva/GNNBAA/GRL_pytorch/embeddings/offline_embedings/mase_fc_{self.emb_size}.npy'
            X = np.load(file)
            return torch.from_numpy(X).to(torch.float32)
        
        
        if self.x_type == 'fc':
            return self._get_fc(subject_name)
        
        
        raise ValueError("Unsupported x_type")

