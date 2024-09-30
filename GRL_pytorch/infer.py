# from config import model_mapping, ROOT_PATH, MATRIX_FILE, DATASET_NAME, X_TYPE, LAMB, HIDDEN_DIMS, BATCH_SIZE, EPOCHS, LR, MODEL, EARLY_STOPPING_PATIENCE, RUNS, RESULTS_PATH, NET_PARAMS, EMB_SIZE, ENCODER_TYPE
from data.dataset import Dataset
from data.dataset_loader import  get_data_loaders, split_dataset_stratified
from tests.test import test_model
import torch
import torch.optim as optim
from utils.plots import save_inference_results
import importlib.util
import sys

ckpt_path = '/home/personal/Documents/2023_2/tesis/GNN-BAA/GRL_pytorch/results/ours/EncoderClassifierFC/mase_10/exp_001/2/ckpt/best_EncoderClassifierFC_Gender_53_10000_0.001.pt'
save_path = '/home/personal/Documents/2023_2/tesis/GNN-BAA/GRL_pytorch/results/ours/EncoderClassifierFC/mase_10/exp_001/2/on_Destrieux/'
config_path = '/home/personal/Documents/2023_2/tesis/GNN-BAA/GRL_pytorch/results/ours/EncoderClassifierFC/mase_10/config.py'

# Especifica el nombre del módulo (puede ser cualquier nombre)
module_name = 'config'
BATCH_SIZE = 19
# Carga el módulo
spec = importlib.util.spec_from_file_location(module_name, config_path)
config = importlib.util.module_from_spec(spec)
sys.modules[module_name] = config
spec.loader.exec_module(config)

dataset = Dataset(root=config.ROOT_PATH, dataset_name = config.DATASET_NAME, x_type = config.X_TYPE, emb_size = config.EMB_SIZE, matrix_file = config.MATRIX_FILE, which_fc = 'ours', num_nodes=167, 
                  fc_paths='/home/personal/Documents/2023_2/tesis/data_processing/data_destrieux/corr_matrices/') # tengo que poner en el config las mismas cualdiades del que entrene, o en su defecto el mismo config
print(dataset[0])
loader, _, _ = get_data_loaders(dataset, None, None, batch_size = config.BATCH_SIZE)

filename = f'{config.MODEL}_{config.DATASET_NAME}_{config.BATCH_SIZE}_{config.EPOCHS}_{config.LR}'

results = []

model_class = config.model_mapping.get(config.MODEL)
model = model_class(hidden_dims = config.HIDDEN_DIMS, num_classes = dataset.num_classes, model_name = config.MODEL) #, net_params = config.NET_PARAMS, encoder_type = config.ENCODER_TYPE)

if torch.cuda.is_available():
    device = torch.device("cuda")  
    model = model.to(device)      
    print("Model and tensors moved to GPU.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")
    
optimizer = optim.Adam(model.parameters(), lr=config.LR)
criterion_recon = torch.nn.MSELoss()
criterion_classif = torch.nn.CrossEntropyLoss(reduction= 'mean')
    
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

test_results = test_model(model, loader, config.LAMB, criterion_recon, criterion_classif)

save_inference_results(test_results, filename, save_path)



