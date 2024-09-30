import numpy as np
import nibabel as nib
from scipy.io import loadmat, savemat
import os
from scipy.stats import pearsonr
from nilearn.connectome import ConnectivityMeasure


def load_atlas_aux(path2atlas):
    ##### Si quisiera correr con Destrieux seria el archivo aparc.a2009s el que usar junto con el
    ##### Destrieux.mat de Zengwhu abajo, creo que funcionaria pero habria que corroborar
    """
    Load a subject-specific desikan label using nibabel.

    Parameters:
    - path2atlas: str, path to the atlas CIFTI file

    Returns:
    - segmentation_atlas: numpy array, atlas segmentation data
    """

    # Check if the atlas file exists
    if not os.path.isfile(path2atlas):
        print("Atlas file does not exist:", path2atlas)
        return None
    
    # Load the CIFTI file
    try:
        cifti_file = nib.load(path2atlas)
    except Exception as e:
        print("Error loading CIFTI file:", e)
        return None

    data = cifti_file.get_fdata()

    # keep the first map
    segmentation_atlas = data[0, :]

    return segmentation_atlas

def get_timeseries(path2fmri, roi_path, path2atlas, subject, timeseries_file = None):
    
        """
        Load functional time series.

        Parameters:
        - path2fmri: str, path to the fMRI .nii file
        - path2atlas: str, path to the atlas file
        - timeseries_file: str, path to save the extracted time series

        Returns:
        - dtseries: 2D numpy array, time series data (num_roi x num_obvs)
        """

        chosen_roi = loadmat(roi_path)
        num_chosen_roi = len(chosen_roi['cortical'][0]) + len(chosen_roi['subcortical'][0])
        segmentation_atlas = load_atlas_aux(path2atlas)  # Define this function as needed

        # Load fMRI data
        fmri_data_nii = nib.load(path2fmri)
        fmri_data = fmri_data_nii.get_fdata()

        ncortex = 59412
        ntotal = 91282
        nTR = fmri_data.shape[0]  # Number of time points
        
        header = fmri_data_nii.header
        brain_models = header.get_index_map(1).brain_models
        start = 0
        end = 0
        start_end_tuplas = []
        for bm_id, bm in enumerate(brain_models):
            start = end
            end = bm.index_offset
            if bm_id>2:
                start_end_tuplas.append((start, end)) 
                # print(bm.brain_structure)
        start_end_tuplas.append((90034, 91282))

        # Initialize dtseries
        dtseries = np.zeros((num_chosen_roi, nTR))

        # Process ROIs
        if 'a2009s' in path2atlas:                              # Destrieux
            for roi_index in range(num_chosen_roi):
                
                indices = np.zeros(fmri_data.shape[1])
        
                if roi_index < 19:  # Subcortical, as an example
                    start, end = start_end_tuplas[roi_index]
                    indices[start:end] = 1
                    indices = indices != 0
                        
                if 19 <= roi_index < 167:  # Cortical
                    
                    roi = chosen_roi['cortical'][0][roi_index-19]
                    indices = np.concatenate([(segmentation_atlas == roi), np.zeros(ntotal - ncortex, dtype=bool)])

                dtseries[roi_index, :] = np.mean(fmri_data[:, indices], axis=1)
        else:                                                    # Desikan
            for roi_index in range(num_chosen_roi):
                
                indices = np.zeros(fmri_data.shape[1])
        
                if roi_index < 19:  # Subcortical, as an example
                    start, end = start_end_tuplas[roi_index]
                    indices[start:end] = 1
                    indices = indices != 0
                        
                if 19 <= roi_index < 87:  # Cortical
                    
                    roi = chosen_roi['cortical'][0][roi_index-19]
                    indices = np.concatenate([(segmentation_atlas == roi), np.zeros(ntotal - ncortex, dtype=bool)])

                dtseries[roi_index, :] = np.mean(fmri_data[:, indices], axis=1)
        
        if timeseries_file != None:
            # Save the time series data
            savemat(timeseries_file, {'dtseries': dtseries})
            print(f'Time_series for subject {subject} saved.')      
            
        return dtseries
    

def concatenate_timeseries(subj_timeseries):
    # Substracts the mean because each measurement is calibrated differently
    normalized_timeseries = [(ts - ts.mean(axis=1, keepdims=True)) / ts.std(axis=1, keepdims=True) for ts in subj_timeseries]
    # Concatenate the normalized timeseries data
    try:
        return np.concatenate(normalized_timeseries, axis=1)
    except Exception as e:
        print(e)
        

def construct_corr(m):
    """
    This function construct correlation matrix from the preprocessed fmri matrix
    Args.

    m (numpy  array): a preprocessed numpy matrix
    return: correlation matrix
    """
    zd_Ytm = (m - np.nanmean(m, axis=1)[0]) / np.nanstd(m, axis=1, ddof=1)[0]
    conn = ConnectivityMeasure(kind = 'correlation')
    fc = conn.fit_transform([m.T])[0]
    zd_fc = conn.fit_transform([zd_Ytm.T])[0]
    fc *= np.tri(*fc.shape)
    np.fill_diagonal(fc, 0)
    # zscored upper triangle
    zd_fc *= 1 - np.tri(*zd_fc.shape, k=-1)
    np.fill_diagonal(zd_fc, 0)
    corr = fc + zd_fc
    return corr
        

def calculate_fc_matrix(timeseries, kind, subject, corr_path = None, task = None):
    """
    Calculate the correlation matrix from the timeseries data.
    """
    num_nodes = timeseries.shape[0]
    fc_matrix_corr = np.zeros((num_nodes, num_nodes))
    fc_matrix_cov = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if i == j:
                fc_matrix_corr[i, j] = 1.0
                fc_matrix_cov[i, j] = 1.0
            else:
                correlation, _ = pearsonr(timeseries[i], timeseries[j])
                fc_matrix_corr[i, j] = fc_matrix_corr[j, i] = correlation
                
                covariance_matrix = np.cov(timeseries[i], timeseries[j])
                covariance = covariance_matrix[0, 1]  # This is the covariance between data[i] and data[j]
                fc_matrix_cov[i, j] = fc_matrix_cov[j, i] = covariance
                
    if kind == 'correlation':
        
        if corr_path is not None:   
            if task is not None:       
                np.save(os.path.join(corr_path, f'fc_{subject}_corr_{task}.npy'), fc_matrix_corr)
            else:
                np.save(os.path.join(corr_path, f'fc_{subject}_corr.npy'), fc_matrix_corr)
            
            # print(f'Correlation matrix for subject {subject} saved.')
                
        return fc_matrix_corr, None
        
        
    elif kind == 'covariance':
        
        if corr_path != None:  
            if task is not None:        
                np.save(os.path.join(corr_path, f'fc_{subject}_cov_{task}.npy'), fc_matrix_cov)
            else:
                np.save(os.path.join(corr_path, f'fc_{subject}_cov.npy'), fc_matrix_cov)
            
            print(f'Covariance matrix for subject {subject} saved.')
                
        return None, fc_matrix_cov
        
        
    elif kind == 'both':
        if corr_path != None: 
            if task is not None:         
                np.save(os.path.join(corr_path, f'fc_{subject}_corr_{task}.npy'), fc_matrix_corr)
                np.save(os.path.join(corr_path, f'fc_{subject}_cov_{task}.npy'), fc_matrix_cov)
            else:
                np.save(os.path.join(corr_path, f'fc_{subject}_corr.npy'), fc_matrix_corr)
                np.save(os.path.join(corr_path, f'fc_{subject}_cov.npy'), fc_matrix_cov)
                
            print(f'Both FC matrices for subject {subject} saved.')
                
        return fc_matrix_corr, fc_matrix_cov
    
    
######### EXPERIMENTS

# def concatenate_timeseries_by_experiment(subj_timeseries):
#     # Substracts the mean because each measurement is calibrated differently
#     # normalized_timeseries = [ts - ts.mean(axis=1, keepdims=True) for ts in subj_timeseries]
#     # Concatenate the normalized timeseries data
#     try:
#         # return np.concatenate(normalized_timeseries, axis=1)
#         ts_1 = np.concatenate((subj_timeseries[0], subj_timeseries[1]), axis = 1)
#         ts_2 = np.concatenate((subj_timeseries[2], subj_timeseries[3]), axis = 1)
#         return (ts_1, ts_2)
#     except Exception as e:
#         print(e)
        
# def concatenate_timeseries_one_by_one(subj_timeseries):
#     # Substracts the mean because each measurement is calibrated differently
#     # normalized_timeseries = [ts - ts.mean(axis=1, keepdims=True) for ts in subj_timeseries]
#     # Concatenate the normalized timeseries data
#     try:
#         return subj_timeseries
#     except Exception as e:
#         print(e)
        
        
# def calculate_fc_matrix_one_by_one(full_timeseries, kind, subject, corr_path = None):
#     """
#     Calculate the correlation matrix from the timeseries data.
#     """
    
#     # ts_1, ts_2 = full_timeseries
    
#     four_corr = []
#     four_cov = []
    
#     for timeseries in full_timeseries:
        
#         num_nodes = timeseries.shape[0]
        
#         fc_matrix_corr = np.zeros((num_nodes, num_nodes))
#         fc_matrix_cov = np.zeros((num_nodes, num_nodes))

#         for i in range(num_nodes):
#             for j in range(i, num_nodes):
#                 if i == j:
#                     fc_matrix_corr[i, j] = 1.0
#                     fc_matrix_cov[i, j] = 1.0
#                 else:
#                     correlation, _ = pearsonr(timeseries[i], timeseries[j])
#                     fc_matrix_corr[i, j] = fc_matrix_corr[j, i] = correlation
                    
#                     covariance_matrix = np.cov(timeseries[i], timeseries[j])
#                     covariance = covariance_matrix[0, 1]  # This is the covariance between data[i] and data[j]
#                     fc_matrix_cov[i, j] = fc_matrix_cov[j, i] = covariance
                    
#         four_corr.append(fc_matrix_corr)
#         four_cov.append(fc_matrix_cov)
        
#     fc_matrix_corr = (four_corr[0] + four_corr[1] + four_corr[2] + four_corr[3])/4
    
#     fc_matrix_cov = (four_cov[0] + four_cov[1] + four_cov[2] + four_cov[3])/4
                
#     if kind == 'correlation':
        
#         if corr_path != None:          
#             np.save(os.path.join(corr_path, f'fc_{subject}_cov.npy'), fc_matrix_corr)
            
#             print(f'Correlation matrix for subject {subject} saved.')
                
#         return fc_matrix_corr, None
        
        
#     elif kind == 'covariance':
        
#         if corr_path != None:          
#             np.save(os.path.join(corr_path, f'fc_{subject}_cov.npy'), fc_matrix_cov)
            
#             print(f'Covariance matrix for subject {subject} saved.')
                
#         return None, fc_matrix_cov
        
        
#     elif kind == 'both':
#         if corr_path != None:          
#             np.save(os.path.join(corr_path, f'fc_{subject}_corr.npy'), fc_matrix_corr)
#             np.save(os.path.join(corr_path, f'fc_{subject}_cov.npy'), fc_matrix_cov)
#             print(f'Both FC matrices for subject {subject} saved.')
                
#         return fc_matrix_corr, fc_matrix_cov 
        
# def calculate_fc_matrix_by_experiment(full_timeseries, kind, subject, corr_path = None):
#     """
#     Calculate the correlation matrix from the timeseries data.
#     """
    
#     # ts_1, ts_2 = full_timeseries
    
#     both_corr = []
#     both_cov = []
    
#     for timeseries in full_timeseries:
        
#         num_nodes = timeseries.shape[0]
        
#         fc_matrix_corr = np.zeros((num_nodes, num_nodes))
#         fc_matrix_cov = np.zeros((num_nodes, num_nodes))

#         for i in range(num_nodes):
#             for j in range(i, num_nodes):
#                 if i == j:
#                     fc_matrix_corr[i, j] = 1.0
#                     fc_matrix_cov[i, j] = 1.0
#                 else:
#                     correlation, _ = pearsonr(timeseries[i], timeseries[j])
#                     fc_matrix_corr[i, j] = fc_matrix_corr[j, i] = correlation
                    
#                     covariance_matrix = np.cov(timeseries[i], timeseries[j])
#                     covariance = covariance_matrix[0, 1]  # This is the covariance between data[i] and data[j]
#                     fc_matrix_cov[i, j] = fc_matrix_cov[j, i] = covariance
                    
#         both_corr.append(fc_matrix_corr)
#         both_cov.append(fc_matrix_cov)
        
#     fc_matrix_corr = (both_corr[0] + both_corr[1])/2
    
#     fc_matrix_cov = (both_cov[0] + both_cov[1])/2
                
#     if kind == 'correlation':
        
#         if corr_path != None:          
#             np.save(os.path.join(corr_path, f'fc_{subject}_cov.npy'), fc_matrix_corr)
            
#             print(f'Correlation matrix for subject {subject} saved.')
                
#         return fc_matrix_corr, None
        
        
#     elif kind == 'covariance':
        
#         if corr_path != None:          
#             np.save(os.path.join(corr_path, f'fc_{subject}_cov.npy'), fc_matrix_cov)
            
#             print(f'Covariance matrix for subject {subject} saved.')
                
#         return None, fc_matrix_cov
        
        
#     elif kind == 'both':
#         if corr_path != None:          
#             np.save(os.path.join(corr_path, f'fc_{subject}_corr.npy'), fc_matrix_corr)
#             np.save(os.path.join(corr_path, f'fc_{subject}_cov.npy'), fc_matrix_cov)
#             print(f'Both FC matrices for subject {subject} saved.')
                
#         return fc_matrix_corr, fc_matrix_cov 
    

######## TO CALL THE FUNCTION BY ITSELF

# dtseries = get_timeseries(path2fmri = '/home/personal/Documents/2023_2/tesis/full_100206/brain_data/100206_complete/100206_3T_rfMRI_REST_fix/100206/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii', 
#                           path2atlas = '/home/personal/Documents/2023_2/tesis/full_100206/brain_data/100206_complete/100206.aparc.32k_fs_LR.dlabel.nii', subject = 100206, timeseries_file = None)
                    