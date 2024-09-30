import torch.optim as optim
import torch
import os
from pathlib import Path

from data.dataset import Dataset
#from data.dataset_modify_order import Dataset
from data.dataset_loader import get_data_loaders, split_dataset_stratified
from training.train import train_and_evaluate
from tests.test import test_model
from utils.utils import check_path
from utils.plots import create_plots_and_save_results, save_average_results
from config import model_mapping, ROOT_PATH, MATRIX_FILE, DATASET_NAME, LAMB, HIDDEN_DIMS, BATCH_SIZE, EPOCHS, LR, MODEL, EARLY_STOPPING_PATIENCE, RUNS, RESULTS_PATH, SAVE_MODEL, FC_PATHS

MODEL = os.getenv('MODEL')
LAMB = float(os.getenv('PARAM'))

dataset = Dataset(root=ROOT_PATH, dataset_name = DATASET_NAME, fc_paths = FC_PATHS, matrix_file= MATRIX_FILE) 
train_dataset, val_dataset, test_dataset = split_dataset_stratified(dataset)
train_loader, val_loader, test_loader = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size = BATCH_SIZE)

general_path = f'{RESULTS_PATH}/{MODEL}/{LAMB}/'
general_path = check_path(general_path, config_path = '/datos/projects/ssilva/GNNBAA/GRL_FC_TEMPORAL/config.py')
filename = f'{MODEL}_{DATASET_NAME}_{BATCH_SIZE}_{EPOCHS}_{LR}_{EARLY_STOPPING_PATIENCE}'

results = []
test_results = []

ckpt_path = None

for run in range(RUNS):
    
    path_to_save = Path(general_path) / f'{run}'
    path_to_best_checkpoint = Path(path_to_save) / 'ckpt' / f'best_{filename}.pt'
    
    print(f'Run {run}')
    model_class = model_mapping.get(MODEL)
    model = model_class(num_classes = dataset.num_classes, hidden_dims = HIDDEN_DIMS, model_name = MODEL)

    if torch.cuda.is_available():
        device = torch.device("cuda:1")  
        model = model.to(device)      
        print("Model and tensors moved to GPU.")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU.")
        
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_recon = torch.nn.MSELoss()
    criterion_classif = torch.nn.CrossEntropyLoss(reduction= 'mean')
    
    result = train_and_evaluate(model, path_to_save, filename, train_loader, val_loader, LAMB, EPOCHS, optimizer, criterion_recon, criterion_classif, EARLY_STOPPING_PATIENCE, path_to_save, SAVE_MODEL)
    
    test_result = test_model(model, path_to_best_checkpoint, test_loader, LAMB, criterion_recon, criterion_classif)
    test_results.append(test_result)
    
    create_plots_and_save_results(result, test_results, filename, path_to_save)
    results.append(result)

save_average_results(results, test_results, filename, general_path)
