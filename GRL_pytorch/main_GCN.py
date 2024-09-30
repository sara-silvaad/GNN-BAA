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
from configs.config_GCN import model_mapping, FC_PATHS, ROOT_PATH, MATRIX_FILE, DATASET_NAME, X_TYPE, LAMB, HIDDEN_DIMS, BATCH_SIZE, EPOCHS, LR, MODEL, EARLY_STOPPING_PATIENCE, RUNS, RESULTS_PATH, EMB_SIZE, POOLING_TYPE, NUM_NODES, NEGS

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

MODEL = os.getenv('MODEL')
NUM_NODES = int(os.getenv('NUM_NODES'))
HIDDEN_DIMS = [NUM_NODES, 32, 16, 8] 
LAMB = float(os.getenv('LAMB'))

negs = os.getenv('NEGS')
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
if MODEL == 'EncoderDecoderSCFC':
    general_path = f'{RESULTS_PATH}/{MODEL}/{X_TYPE}/{LAMB}/{NUM_NODES}_nodes_{NEGS}_negs/'
else:
    general_path = f'{RESULTS_PATH}/{MODEL}/{X_TYPE}/{NUM_NODES}_nodes_{NEGS}_negs/'
    
general_path = check_path(general_path, config_path = '/home/personal/Documents/2023_2/tesis/GNN-BAA/GRL_pytorch/configs/config_GCN.py')
filename = f'{MODEL}_{DATASET_NAME}_{BATCH_SIZE}_{EPOCHS}_{LR}'

# ckpt_path = '/home/personal/Documents/2023_2/tesis/GNN-BAA/GRL_pytorch/results/MAX/87_nodes/EncoderClassifierSCTransformerEmbedding/31/5_aligned/exp_004/0/ckpt/last_EncoderClassifierSCTransformerEmbedding_Gender_64_10000_0.001.pt'
ckpt_path = None

results = []
test_results = []

# Runs
for run in range(RUNS):
    path_to_save = Path(general_path) / f'{run}'
    path_to_best_checkpoint = Path(path_to_save) / 'ckpt' / f'best_{filename}.pt'
    
    print(f'Run {run}')
    model_class = model_mapping.get(MODEL)
    model = model_class(hidden_dims = HIDDEN_DIMS, num_classes = dataset.num_classes, model_name = MODEL, pooling_type = POOLING_TYPE, num_nodes=NUM_NODES, negs=NEGS)
    
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
    
    result = train_and_evaluate(model, train_loader, val_loader, EPOCHS, optimizer, criterion_recon, criterion_classif, EARLY_STOPPING_PATIENCE, path_to_save, filename, ckpt_path = ckpt_path, lamb = LAMB)
    test_result = test_model(model, path_to_best_checkpoint, test_loader, criterion_recon, criterion_classif, lamb = LAMB)
    
    create_plots_and_save_results(result, test_result, filename, path_to_save)
    
    results.append(result)
    test_results.append(test_result)

save_average_results(results, test_results, filename, general_path)

