import os
import torch
import numpy as np
import pandas as pd
import scipy.io
from torch_geometric.data import InMemoryDataset, Data
import hashlib
import torch
from torch_geometric.utils import dense_to_sparse


class Dataset(InMemoryDataset):
    def __init__(self, root, dataset_name, x_type = 'one_hot', negs = False, num_nodes = 87, transform=None, pre_transform=None, matrix_file=None, fc_paths =None , labels_path = '/datos/projects/ssilva/GNNBAA/data/original_data/HCP_behavioral.csv'):
        self.matrix_file = matrix_file
        self.x_type = x_type
        self.dataset_name = dataset_name
        self.negs = negs
        self.fc_paths = fc_paths
        self.num_nodes = num_nodes
        self.labels_path = labels_path
        self.unique_id = self._generate_unique_id(dataset_name, matrix_file, negs, num_nodes, x_type)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.matrix_file]

    @property
    def processed_file_names(self):
        return [f'data_{self.unique_id}.pt']

    def _generate_unique_id(self, dataset_name, matrix_file, negs, num_nodes, x_type):
        params_str = f"{dataset_name}-{matrix_file}-{negs}-{num_nodes}-{x_type}-FC_temporal"
        return hashlib.md5(params_str.encode()).hexdigest()

    def download(self):
        pass  # Implement if necessary

    def process(self):
        label_data = self._load_label_data()
        mat = scipy.io.loadmat(self.raw_paths[0])
        subject_list = mat['subject_list'].squeeze()
        data_list = [self._process_subject(subject_name, label_data) for idx, subject_name in enumerate(subject_list) if subject_name in label_data]
        self.data, self.slices = self.collate([d for d in data_list if d is not None])
        torch.save((self.data, self.slices), self.processed_paths[0])

    def _load_label_data(self):
        label_data = pd.read_csv(self.labels_path)
        return dict(zip(label_data['Subject'], zip(label_data['Gender'], label_data['Age'])))

    def _process_subject(self, subject_name, label_data):
        print(f'procesing subject {subject_name}')
        try:
            fc, corr = self._get_multiple_fc(subject_name)
            edge_index_fc, edge_weight_fc = [], []
            
            for fc_ in fc:
                edge_index, edge_weight = self._convert_to_edge_index(fc_)
                edge_index_fc.append(edge_index)
                edge_weight_fc.append(edge_weight)
                
            edge_index_corr, edge_weight_corr = self._convert_to_edge_index(corr)

            # features
            x = self._get_x(fc, subject_name)

            # labels
            y = self._get_label(subject_name, label_data)

            if y is not None:
                return Data(x=x, fc = fc, corr = corr, edge_index_fc = edge_index_fc, edge_weight_fc = edge_weight_fc, edge_index_corr = edge_index_corr, edge_weight_corr = edge_weight_corr, y = y)
        except Exception as e:
            print(f"Error processing subject {subject_name}: {e}")
        return None

    def _get_x(self, sc, subject_name):
        if self.x_type == 'one_hot':
            return torch.eye(self.num_nodes, dtype=torch.float)
        

    def _get_multiple_fc(self, subject_name):
        # self.fc_paths es una variable nueva con el path a la carpeta que sea que quieras usar donde estan las fc, ponele donde hay una sola o varias aumentadas

        path_to_subject_fc = os.path.join(f'{self.fc_paths}augmented_data/{subject_name}')
        fc_list = [np.load(os.path.join(path_to_subject_fc, fc)) for fc in os.listdir(path_to_subject_fc) if fc.endswith('corr.npy')]
        
        # ahi ya tenes una lista con todas las fc que haya en la carpeta

        # Our matrices have the first 19 subcortical nodes, le hacemos el preprocesamiento a cada una, pero si quiero mantener todo no me importa nada
        fc_list_ = []
        for fc in fc_list:
            if self.num_nodes == 87:
                fc = fc[:, :].astype(np.float32)
            else:
                fc = fc[19:, 19:].astype(np.float32)
            if self.negs == False:
                fc = np.where(fc < 0, 0, fc)
            fc_list_.append(torch.tensor(fc, dtype = torch.float32))

        fc_corr = np.load(f'{self.fc_paths}corr_matrices/{subject_name}/fc_{subject_name}_corr.npy')
        fc_corr = np.where(fc_corr < 0, 0 , fc_corr)
                            
        return fc_list_, torch.tensor(fc_corr, dtype=torch.float)
    


    def _convert_to_edge_index(self, M):

        edge_index = M.nonzero(as_tuple=False)
        edge_index = edge_index.t().contiguous()
        edge_weights = M[edge_index[0], edge_index[1]].view(-1, 1)
                
        return edge_index, edge_weights


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

