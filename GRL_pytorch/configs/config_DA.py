from models.GCN_encoder_decoder_classifier_MP_DA import EncoderClassifierSCSuperNode, EncoderClassifierSCSpurious, EncoderClassifierSCMaskAttributes, EncoderClassifierSCZoneMaskAttributes

# data
LABELS_PATH = '/datos/projects/ssilva/GNNBAA/data/original_data/HCP_behavioral.csv'
ROOT_PATH = '/datos/projects/ssilva/GNNBAA/data/processed_datasets'
MATRIX_FILE = '/datos/projects/ssilva/GNNBAA/data/original_data/scs_desikan.mat'
FC_PATHS = '/datos/projects/ssilva/GNNBAA/data/87_nodes_downloaded_on_imerl/corr_matrices/'

DATASET_NAME = 'Gender'

RESIDUAL_CONNECTION = False

MODEL = 'EncoderClassifierSCZoneMaskAttributes'

model_mapping = {
    'EncoderClassifierSCZoneMaskAttributes': EncoderClassifierSCZoneMaskAttributes,
    'EncoderClassifierSCMaskAttributes': EncoderClassifierSCMaskAttributes, # N = 40, 87
    'EncoderClassifierSCSpurious': EncoderClassifierSCSpurious, # intensity = 10
    'EncoderClassifierSCSuperNode': EncoderClassifierSCSuperNode, # intensity_super_node = 100, 200, 500, 1000
}

# model
HIDDEN_DIMS = [87, 32, 16, 8] # con veps el primer numero es el numero de veps , con one-hot es 68 (u87)

# training
EPOCHS = 10000
LR = 0.001
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 500
RUNS = 3

#results path
RESULTS_PATH = '/datos/projects/ssilva/GNNBAA/GRL_pytorch/results'
