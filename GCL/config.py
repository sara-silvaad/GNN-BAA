LABELS_PATH = '/datos/projects/ssilva/GNNBAA/data/original_data/HCP_behavioral.csv'
ROOT_PATH = '/datos/projects/ssilva/GNNBAA/data/processed_datasets'
MATRIX_FILE = '/datos/projects/ssilva/GNNBAA/data/original_data/scs_desikan.mat'
FC_PATHS = '/datos/projects/ssilva/GNNBAA/data/87_nodes_downloaded_on_imerl/corr_matrices/'

DATASET_NAME = 'Gender'
NUM_CLASSES = 2

TAU = None

# model
DATA_AUG = 'featureMasking_edgeDropping' # o region
HIDDEN_DIMS = [87, 32, 16, 8] 

# training
# pre training
EPOCHS_pre = 10000
LR_pre = 0.001
BATCH_SIZE_pre = 128
EARLY_STOPPING_PATIENCE_pre = 500
RUNS_pre = 1
SAVE_MODEL_pre = True
LAMBDA_VAL = 1

#fine tune
EPOCHS_ft = 200
EPOCHS_all_ft = 100
LR_ft = 0.001
BATCH_SIZE_ft = 32
EARLY_STOPPING_PATIENCE_ft = 100
RUNS_ft = 5
SAVE_MODEL_ft = True
SELECTED_RUN = 0

#results path pretrained
SAVED_MODELS_PATH_pre = '/datos/projects/ssilva/GNNBAA/GCL/saved_models/PretrainedGCL'
RESULTS_PATH_pre = '/datos/projects/ssilva/GNNBAA/GCL/results/PretrainedGCL'

#results path ft
SAVED_MODELS_PATH_ft = '/datos/projects/ssilva/GNNBAA/GCL/saved_models/FinetuneGCL'
RESULTS_PATH_ft = '/datos/projects/ssilva/GNNBAA/GCL/results/FinetuneGCL'

