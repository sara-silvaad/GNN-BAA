import torch.optim as optim
import torch
import os

from pathlib import Path
from data.dataset import Dataset

from data.dataset_loader import  get_data_loaders, split_dataset_stratified
from training.train import train_and_evaluate
from tests.test import test_model
from utils.plots import create_plots_and_save_results, save_average_results
from utils.utils import check_path
from configs.config_nn import model_mapping, FC_PATHS, ROOT_PATH, MATRIX_FILE, DATASET_NAME, X_TYPE, BATCH_SIZE, EPOCHS, LR, MODEL, EARLY_STOPPING_PATIENCE, RUNS, RESULTS_PATH, EMB_SIZE, NUM_NODES, NEGS

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

MODEL = os.getenv('MODEL')
NUM_NODES = int(os.getenv('NUM_NODES'))

negs = os.getenv('NUM_NODES')
if negs == 'False':
    NEGS = False
else:
    NEGS = True
    
# Data
dataset = Dataset(root=ROOT_PATH, dataset_name = DATASET_NAME, x_type = X_TYPE, emb_size=EMB_SIZE, 
                  fc_paths = FC_PATHS, num_nodes=NUM_NODES, negs = NEGS, matrix_file= MATRIX_FILE) 

train_dataset, val_dataset, test_dataset = split_dataset_stratified(dataset)
train_loader, val_loader, test_loader = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size = BATCH_SIZE)

# Solve path saving 
general_path = f'{RESULTS_PATH}/{MODEL}/{NUM_NODES}_nodes_{NEGS}_negs/'

general_path = check_path(general_path, config_path = '/home/personal/Documents/2023_2/tesis/GNN-BAA/GRL_pytorch/configs/config_nn.py')
filename = f'{MODEL}_{DATASET_NAME}_{BATCH_SIZE}_{EPOCHS}_{LR}'

# ckpt_path = '/home/personal/Documents/2023_2/tesis/GNN-BAA/GRL_pytorch/results/MAX/87_nodes/EncoderClassifierSCTransformerEmbedding/31/5_aligned/exp_004/0/ckpt/last_EncoderClassifierSCTransformerEmbedding_Gender_64_10000_0.001.pt'
ckpt_path = None

results = []
test_results = []

# Runs
for run in range(RUNS):
    path_to_save = Path(general_path) / f'{run}'
    
    print(f'Run {run}')
    model_class = model_mapping.get(MODEL)
    model = model_class(num_nodes = dataset.num_nodes, num_classes = dataset.num_classes)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")  
        model = model.to(device)      
        print("Model and tensors moved to GPU.")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU.")
        
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_recon = torch.nn.MSELoss()
    criterion_classif = torch.nn.CrossEntropyLoss(reduction= 'mean')
    
    result = train_and_evaluate(model, train_loader, val_loader, LAMB, EPOCHS, optimizer, criterion_recon, criterion_classif, EARLY_STOPPING_PATIENCE, path_to_save, filename, ckpt_path = ckpt_path)
    test_result = test_model(model, path_to_best_checkpoint, test_dataset, LAMB, criterion_recon, criterion_classif)
    
    create_plots_and_save_results(result, test_result, filename, path_to_save)
    results.append(result)
    test_results.append(test_result)

save_average_results(results, test_results, filename, general_path)
