import torch.optim as optim
import torch
import os
from utils.set_optimizer import set_optimizer
from data.dataset import Dataset
from data.dataset_loader import get_data_loaders, split_dataset_stratified
from training.train import train_and_evaluate_ft
from tests.test import test_ft
from models.GCN_encoder_decoder_classifier import load_pretrained_model, FineTuneClassifier
from utils.plots import create_plots_and_save_results, save_average_results, solve_paths_pre, solve_paths_ft
from config import ROOT_PATH, MATRIX_FILE, DATASET_NAME, HIDDEN_DIMS, BATCH_SIZE_ft, EPOCHS_ft, LR_ft, EARLY_STOPPING_PATIENCE_ft, RUNS_ft, DATA_AUG, EPOCHS_all_ft, EPOCHS_pre, LR_pre
from config import SAVED_MODELS_PATH_ft, RESULTS_PATH_ft, SELECTED_RUN, FC_PATHS, SAVED_MODELS_PATH_pre, RESULTS_PATH_pre

MODEL = 'FineTuneClassifier'
LAMBDA_VAL = None

DATA_AUG = os.getenv('DATA_AUG')
TAU = float(os.getenv('TAU'))
norm = os.getenv('NORM')

if norm == 'True':
  NORM = True
else:
  NORM = False
  
HIDDEN_DIM = int(os.getenv('HIDDEN_DIM'))

# SEt the paths for the finetune model
FT_MODEL_PATH, FT_RESULTS_PATH = solve_paths_pre(SAVED_MODELS_PATH_ft, RESULTS_PATH_ft, DATA_AUG, TAU)
ft_model_name = f'{MODEL}_{DATA_AUG}_{BATCH_SIZE_ft}_{EPOCHS_ft}_{LR_ft}_{EARLY_STOPPING_PATIENCE_ft}_{NORM}_HD{HIDDEN_DIM}_concatenate_correct_2'#_smaller_encoder'

# About the pretrained model
PRE_MODEL_PATH, _ = solve_paths_pre(SAVED_MODELS_PATH_pre, RESULTS_PATH_pre, DATA_AUG, TAU)
pre_model_name = f'{SELECTED_RUN}_{DATA_AUG}_{EPOCHS_pre}_{LR_pre}_concatenation_best_model.pt'

dataset = Dataset(root=ROOT_PATH, dataset_name = DATASET_NAME, matrix_file= MATRIX_FILE, fc_paths = FC_PATHS)
train_dataset, val_dataset, test_dataset = split_dataset_stratified(dataset)
train_loader, val_loader, test_loader = get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size = BATCH_SIZE_ft)

results = []
test_results = []

for run in range(RUNS_ft):
    print(f'Run {run}')

    encoder = load_pretrained_model(f"{PRE_MODEL_PATH}/{pre_model_name}", HIDDEN_DIMS, dataset.num_classes, DATA_AUG, pooling_type = 'concatenate')
    # classifier = FineTuneClassifier(input_dim=sum(HIDDEN_DIMS[1:]), hidden_dim = HIDDEN_DIM, normalize = NORM)
    classifier = FineTuneClassifier(input_dim= 87*sum(HIDDEN_DIMS[1:]), hidden_dim = HIDDEN_DIM, normalize = NORM)

    criterion_classif = torch.nn.CrossEntropyLoss(reduction = 'mean')

    # train and test
    result, best_model = train_and_evaluate_ft(encoder, classifier, train_loader, val_loader, EPOCHS_ft, LR_ft, criterion_classif, EARLY_STOPPING_PATIENCE_ft, EPOCHS_all_ft, spurious_connetions=None)
    test_result = test_ft(encoder, classifier, test_loader, criterion_classif, spurious_connetions=None)
    results.append(result)
    test_results.append(test_result)

    # save results and model
    create_plots_and_save_results(result, test_result, f'{run}_{ft_model_name}', best_model, FT_RESULTS_PATH, FT_MODEL_PATH)
    
# save average results
save_average_results(results, test_results, ft_model_name, FT_RESULTS_PATH)
