from models.GCN_encoder_decoder_classifier import EncoderDecoderSCFC, EncoderClassifierSC, EncoderClassifierFC, EncoderClassifierSCSPE, EncoderClassifierFCSPE

# data
LABELS_PATH = '/datos/projects/ssilva/GNNBAA/data/original_data/HCP_behavioral.csv'
ROOT_PATH = '/datos/projects/ssilva/GNNBAA/data/processed_datasets'
MATRIX_FILE = '/datos/projects/ssilva/GNNBAA/data/original_data/scs_desikan.mat'
FC_PATHS = '/datos/projects/ssilva/GNNBAA/data/87_nodes_downloaded_on_imerl/corr_matrices'

DATASET_NAME = 'Gender'

X_TYPE = 'one_hot' 
EMB_SIZE = 87

MODEL = 'EncoderClassifierSC' 

NUM_NODES = 87
NEGS = False

POOLING_TYPE = 'ave'
 
model_mapping = {
    'EncoderDecoderSCFC': EncoderDecoderSCFC,
    'EncoderClassifierSC': EncoderClassifierSC,
    'EncoderClassifierFC': EncoderClassifierFC,
    'EncoderClassifierSCSPE': EncoderClassifierSCSPE,
    'EncoderClassifierFCSPE': EncoderClassifierFCSPE,
}

# Only for EncoderDecoderSCFC
LAMB = 0.6

HIDDEN_DIMS = [EMB_SIZE, 32, 16, 8] 

# training
EPOCHS = 10000
LR = 0.001
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 300
RUNS = 7

#results path
RESULTS_PATH = '/datos/projects/ssilva/GNNBAA/GRL_pytorch/results'