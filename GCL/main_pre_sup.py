import torch.optim as optim
import os
from data.dataset import Dataset
from data.dataset_loader import get_data_loaders, split_dataset_stratified
from training.train import train_and_evaluate_pre
from tests.test import test_pre
from models.GCN_encoder_decoder_classifier import GCL_FeatureMaskingEdgeDropping, GCL_region, GCL_FeatureMaskingEdgeDroppingDecoder
from utils.plots import create_plots_and_save_results, save_average_results, solve_paths_pre
from config import ROOT_PATH, MATRIX_FILE, DATASET_NAME, HIDDEN_DIMS, BATCH_SIZE_pre, EPOCHS_pre, LR_pre, EARLY_STOPPING_PATIENCE_pre, RUNS_pre 
from config import LAMBDA_VAL, FC_PATHS, DATA_AUG, SAVED_MODELS_PATH_pre, RESULTS_PATH_pre, TAU
import torch

LAMBDA_VAL = None

DATA_AUG = os.getenv('DATA_AUG')
TAU = float(os.getenv('TAU'))
norm = os.getenv('NORM')

if norm == 'True':
  NORM = True
else:
  NORM = False
  
HIDDEN_DIM = int(os.getenv('HIDDEN_DIM'))


PRE_MODEL_PATH, PRE_RESULTS_PATH = solve_paths_pre(SAVED_MODELS_PATH_pre, RESULTS_PATH_pre, DATA_AUG, TAU)
pre_model_name = f'{DATA_AUG}_{EPOCHS_pre}_{LR_pre}_spurious_connection'

dataset = Dataset(root=ROOT_PATH, dataset_name = DATASET_NAME, matrix_file= MATRIX_FILE, 
                  fc_paths = FC_PATHS)

train_dataset, val_dataset, test_dataset = split_dataset_stratified(dataset)
train_loader, val_loader, test_loader = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size = BATCH_SIZE_pre)

results = []
test_results = []
for run in range(RUNS_pre):
    print(f'Run {run}')

    # define model
    if DATA_AUG == 'featureMasking_edgeDropping':
      model = GCL_FeatureMaskingEdgeDropping(hidden_dims = HIDDEN_DIMS, num_classes = dataset.num_classes, spurious=10, pooling_type='concatenate')
    elif DATA_AUG == 'region':
      model = GCL_region(hidden_dims = HIDDEN_DIMS, num_classes = dataset.num_classes)
    elif DATA_AUG == 'featureMasking_edgeDropping_Decoder':
      model = GCL_FeatureMaskingEdgeDroppingDecoder(hidden_dims = HIDDEN_DIMS, num_classes = dataset.num_classes)

    if torch.cuda.is_available():
        device = torch.device("cuda")  
        model = model.to(device)      
        print("Model and tensors moved to GPU.")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU.")
    
    optimizer = optim.Adam(model.parameters(), lr=LR_pre)

    criterion_recon = torch.nn.MSELoss()
    # criterion_recon = None
    
    result, best_model = train_and_evaluate_pre(model, train_loader, val_loader, EPOCHS_pre, optimizer, EARLY_STOPPING_PATIENCE_pre, 
                                                lambda_val=LAMBDA_VAL, data_aug = DATA_AUG, tau=TAU, loss_recon = criterion_recon)
    
    results.append(result)

    test_result = test_pre(model, test_loader, lambda_val=LAMBDA_VAL, data_aug = DATA_AUG, tau=TAU, loss_recon = criterion_recon)
    test_results.append(test_result)

    create_plots_and_save_results(result, test_result, f'{run}_{pre_model_name}', best_model, PRE_RESULTS_PATH, PRE_MODEL_PATH)
    

# save average pretrained
save_average_results(results, test_results, pre_model_name, PRE_RESULTS_PATH)
