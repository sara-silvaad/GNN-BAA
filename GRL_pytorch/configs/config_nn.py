from models.fully_connected import FullyConnectedFC, FullyConnectedSC, FullyConnectedSCFC

# data
LABELS_PATH = '/datos/projects/ssilva/GNNBAA/data/original_data/HCP_behavioral.csv'
ROOT_PATH = '/datos/projects/ssilva/GNNBAA/data/processed_datasets'
MATRIX_FILE = '/datos/projects/ssilva/GNNBAA/data/original_data/scs_desikan.mat'
FC_PATHS = '/datos/projects/ssilva/GNNBAA/data/87_nodes_downloaded_on_imerl/corr_matrices/'

DATASET_NAME = 'Gender'

X_TYPE = 'one_hot' 
EMB_SIZE = 87

NUM_NODES = 87
NEGS = False
 
MODEL = 'EncoderClassifierSC'
 
model_mapping = {
    'FullyConnectedFC': FullyConnectedFC,
    'FullyConnectedSC': FullyConnectedSC,
    'FullyConnectedSCFC': FullyConnectedSCFC,
}


# training
EPOCHS = 10000
LR = 0.001
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 50
RUNS = 10

#results path
RESULTS_PATH = '/datos/projects/ssilva/GNNBAA/GCL/results'