from models.GCN_encoder_decoder_classifier import EncoderClassifierFCRandomTemporal, EncoderDecoderFCFCTemporal
# data

LABELS_PATH = '/datos/projects/ssilva/GNNBAA/data/original_data/HCP_behavioral.csv'
ROOT_PATH = '/datos/projects/ssilva/GNNBAA/data/processed_datasets'
MATRIX_FILE = '/datos/projects/ssilva/GNNBAA/data/original_data/scs_desikan.mat'
FC_PATHS = '/datos/projects/ssilva/GNNBAA/data/87_nodes_downloaded_on_imerl/'

DATASET_NAME = 'Gender'
NUM_CLASSES = 2

MODEL = 'EncoderDecoderFCFCTemporal'
#MODEL = 'EncoderClassifierFCRandomTemporal'

model_mapping = {
    'EncoderDecoderFCFCTemporal': EncoderDecoderFCFCTemporal,
    'EncoderClassifierFCRandomTemporal': EncoderClassifierFCRandomTemporal,
}

# model
LAMB = 0.5
HIDDEN_DIMS = [87, 32, 16, 8] # con veps el primer numero es el numero de veps , con one-hot es 68 (u87)

# training
EPOCHS = 10000
LR = 0.001
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 500
RUNS = 5

#results path
RESULTS_PATH = '/datos/projects/ssilva/GNNBAA/GRL_FC_TEMPORAL/results'
SAVE_MODEL = True

