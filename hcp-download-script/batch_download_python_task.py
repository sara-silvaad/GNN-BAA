#useful resource for aws:
# AWS S3 high level api:
#   https://docs.aws.amazon.com/cli/latest/userguide/cli-services-s3-commands.html

# search for CHANGE for locations of where to customize for new files

import os
import time
import subprocess
from pathlib import Path
from scipy.io import savemat
import shutil
from utils import calculate_fc_matrix, concatenate_timeseries, get_timeseries


def report_missing_files(patients_missing_file):
    
    lr_gambling_miss = patients_missing_file["LR_gambling"]
    rl_gambling_miss = patients_missing_file["RL_gambling"]
    both_gambling_miss = lr_gambling_miss.intersection(rl_gambling_miss)
    lr_len, rl_len, both_len= len(lr_gambling_miss), len(rl_gambling_miss), len(both_gambling_miss)
    print(f"GAMBLING missing:    lr {lr_len} | rl {rl_len} | both {both_len}")

    lr_motor_miss = patients_missing_file["LR_motor"]
    rl_motor_miss = patients_missing_file["RL_motor"]
    both_motor_miss = lr_motor_miss.intersection(rl_motor_miss)
    lr_len,rl_len,both_len=len(lr_motor_miss),len(rl_motor_miss), len(both_motor_miss)
    print(f"MOTOR missing: lr {lr_len} | rl {rl_len} | both {both_len}")
    
    lr_language_miss = patients_missing_file["LR_language"]
    rl_language_miss = patients_missing_file["RL_language"]
    both_language_miss = lr_language_miss.intersection(rl_language_miss)
    lr_len,rl_len,both_len=len(lr_language_miss),len(rl_language_miss), len(both_language_miss)
    print(f"LANGUAGE missing: lr {lr_len} | rl {rl_len} | both {both_len}")
    
    lr_emotion_miss = patients_missing_file["LR_emotion"]
    rl_emotion_miss = patients_missing_file["RL_emotion"]
    both_emotion_miss = lr_emotion_miss.intersection(rl_emotion_miss)
    lr_len,rl_len,both_len=len(lr_motor_miss),len(rl_emotion_miss), len(both_emotion_miss)
    print(f"EMOTION missing: lr {lr_len} | rl {rl_len} | both {both_len}")
    

def end_of_process_sumary(subject_list, patients_missing_file):
    
    #End of Download Summary: CHANGE - include new files here
    print(f"======END OF DOWNLOAD=======")
    print(f"Total Patients: {len(subject_list)}")
    lr_gambling_miss = patients_missing_file["LR_gambling"]
    rl_gambling_miss = patients_missing_file["RL_gambling"]
    both_gambling_miss = lr_gambling_miss.intersection(rl_gambling_miss)
    lr_len, rl_len, both_len= len(lr_gambling_miss), len(rl_gambling_miss), len(both_gambling_miss)
    print(f"GAMBLING missing:    lr {lr_len} | rl {rl_len} | both {both_len}")

    lr_motor_miss = patients_missing_file["LR_motor"]
    rl_motor_miss = patients_missing_file["RL_motor"]
    both_motor_miss = lr_motor_miss.intersection(rl_motor_miss)
    lr_len,rl_len,both_len=len(lr_motor_miss),len(rl_motor_miss), len(both_motor_miss)
    print(f"MOTOR missing: lr {lr_len} | rl {rl_len} | both {both_len}")
    
    lr_language_miss = patients_missing_file["LR_language"]
    rl_language_miss = patients_missing_file["RL_language"]
    both_language_miss = lr_language_miss.intersection(rl_language_miss)
    lr_len,rl_len,both_len=len(lr_language_miss),len(rl_language_miss), len(both_language_miss)
    print(f"LANGUAGE missing: lr {lr_len} | rl {rl_len} | both {both_len}")
    
    lr_emotion_miss = patients_missing_file["LR_emotion"]
    rl_emotion_miss = patients_missing_file["RL_emotion"]
    both_emotion_miss = lr_emotion_miss.intersection(rl_emotion_miss)
    lr_len,rl_len,both_len=len(lr_motor_miss),len(rl_emotion_miss), len(both_emotion_miss)
    print(f"EMOTION missing: lr {lr_len} | rl {rl_len} | both {both_len}")

    print(f"DESIKAN missing: {len(patients_missing_file['Desikan_aparc32k'])}")
    print(f"DESTRIE missing: {len(patients_missing_file['Destrieux_aparc32k'])}")
    print(f"======END OF DOWNLOAD=======")

    lr_no_msmall_miss = [int(a) for a in lr_no_msmall_miss]
    rl_no_msmall_miss = [int(a) for a in rl_no_msmall_miss]
    both_no_msmall_miss = [int(a) for a in both_no_msmall_miss]


def get_file_names(r_num_all):
    
    # Path example
    #"HCP_1200/996782/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii"

    
    HCP_paths = {r_num:{} for r_num in r_num_all}
    
    for r_num in r_num_all:
    
        HCP_paths[r_num]['lr_gambling'] = f"tfMRI_GAMBLING_LR_Atlas.dtseries.nii"
        HCP_paths[r_num]['rl_gambling'] = f"tfMRI_GAMBLING_RL_Atlas.dtseries.nii"
        
        HCP_paths[r_num]['lr_motor'] = f"tfMRI_MOTOR_LR_Atlas.dtseries.nii"
        HCP_paths[r_num]['rl_motor'] = f"tfMRI_MOTOR_RL_Atlas.dtseries.nii"
        
        HCP_paths[r_num]['lr_language'] = f"tfMRI_LANGUAGE_LR_Atlas.dtseries.nii"
        HCP_paths[r_num]['rl_language'] = f"tfMRI_LANGUAGE_RL_Atlas.dtseries.nii"
        
        HCP_paths[r_num]['lr_emotion'] = f"tfMRI_EMOTION_LR_Atlas.dtseries.nii"
        HCP_paths[r_num]['rl_emotion'] = f"tfMRI_EMOTION_RL_Atlas.dtseries.nii"

        HCP_paths[r_num]['lr_gambling_path'] = f"/MNINonLinear/Results/tfMRI_GAMBLING_LR/" + HCP_paths[r_num]['lr_gambling']
        HCP_paths[r_num]['rl_gambling_path'] = f"/MNINonLinear/Results/tfMRI_GAMBLING_RL/" + HCP_paths[r_num]['rl_gambling']
        
        HCP_paths[r_num]['lr_motor_path'] = f"/MNINonLinear/Results/tfMRI_MOTOR_LR/" + HCP_paths[r_num]['lr_motor']
        HCP_paths[r_num]['rl_motor_path'] = f"/MNINonLinear/Results/tfMRI_MOTOR_RL/" + HCP_paths[r_num]['rl_motor']
        
        HCP_paths[r_num]['lr_language_path'] = f"/MNINonLinear/Results/tfMRI_LANGUAGE_LR/" + HCP_paths[r_num]['lr_language']
        HCP_paths[r_num]['rl_language_path'] = f"/MNINonLinear/Results/tfMRI_LANGUAGE_RL/" + HCP_paths[r_num]['rl_language']
        
        HCP_paths[r_num]['lr_emotion_path'] = f"/MNINonLinear/Results/tfMRI_EMOTION_LR/" + HCP_paths[r_num]['lr_emotion']
        HCP_paths[r_num]['rl_emotion_path'] = f"/MNINonLinear/Results/tfMRI_EMOTION_RL/" + HCP_paths[r_num]['rl_emotion']

    # Atlas
    #LR stands for Left and Right hemisphere not Left to Right scanning (as in the dtseries.nii)
    HCP_paths['atlas_file_pre']  = "/MNINonLinear/fsaverage_LR32k/"
    HCP_paths['destrieux_atlas_file_post'] = ".aparc.a2009s.32k_fs_LR.dlabel.nii"
    HCP_paths['desikan_atlas_file_post'] = ".aparc.32k_fs_LR.dlabel.nii"

    #paths on aws machine
    HCP_paths['path2HCP_1200'] = "/hcp-openaccess/HCP_1200/"
    
    return HCP_paths



def list_files(patient_id, HCP_paths, subject_dir, r_num_all):
    
    patient_id = str(patient_id)

    #list of dictionaries representing each file
    files = []

    for r_num in r_num_all:

        #gambling
        #LR
        hcp_path = HCP_paths['path2HCP_1200'] + patient_id + HCP_paths[r_num]['lr_gambling_path']
        readable_name = "LR_gambling"
        local_path = subject_dir + "/" + HCP_paths[r_num]['lr_gambling']
        files.append({"hcp_path": hcp_path, "readable_name": readable_name, "local_path": local_path})
        #RL
        hcp_path = HCP_paths['path2HCP_1200'] + patient_id + HCP_paths[r_num]['rl_gambling_path']
        readable_name = "RL_gambling"
        local_path = subject_dir + "/" + HCP_paths[r_num]['rl_gambling']
        files.append({"hcp_path": hcp_path, "readable_name": readable_name, "local_path": local_path})
        
        #motor
        #LR
        hcp_path = HCP_paths['path2HCP_1200'] + patient_id + HCP_paths[r_num]['lr_motor_path']
        readable_name = "LR_motor"
        local_path = subject_dir + "/" + HCP_paths[r_num]['lr_motor']
        files.append({"hcp_path": hcp_path, "readable_name": readable_name, "local_path": local_path})
        #RL
        hcp_path = HCP_paths['path2HCP_1200'] + patient_id + HCP_paths[r_num]['rl_motor_path']
        readable_name = "RL_motor"
        local_path = subject_dir + "/" + HCP_paths[r_num]['rl_motor']
        files.append({"hcp_path": hcp_path, "readable_name": readable_name, "local_path": local_path})
        
        #language
        #LR
        hcp_path = HCP_paths['path2HCP_1200'] + patient_id + HCP_paths[r_num]['lr_language_path']
        readable_name = "LR_language"
        local_path = subject_dir + "/" + HCP_paths[r_num]['lr_language']
        files.append({"hcp_path": hcp_path, "readable_name": readable_name, "local_path": local_path})
        #RL
        hcp_path = HCP_paths['path2HCP_1200'] + patient_id + HCP_paths[r_num]['rl_language_path']
        readable_name = "RL_language"
        local_path = subject_dir + "/" + HCP_paths[r_num]['rl_language']
        files.append({"hcp_path": hcp_path, "readable_name": readable_name, "local_path": local_path})
        
        #emotion
        #LR
        hcp_path = HCP_paths['path2HCP_1200'] + patient_id + HCP_paths[r_num]['lr_emotion_path']
        readable_name = "LR_emotion"
        local_path = subject_dir + "/" + HCP_paths[r_num]['lr_emotion']
        files.append({"hcp_path": hcp_path, "readable_name": readable_name, "local_path": local_path})
        #RL
        hcp_path = HCP_paths['path2HCP_1200'] + patient_id + HCP_paths[r_num]['rl_emotion_path']
        readable_name = "RL_emotion"
        local_path = subject_dir + "/" + HCP_paths[r_num]['rl_emotion']
        files.append({"hcp_path": hcp_path, "readable_name": readable_name, "local_path": local_path})

    #Destrieux atlas
    hcp_path = HCP_paths['path2HCP_1200'] + patient_id + HCP_paths['atlas_file_pre'] + patient_id + HCP_paths['destrieux_atlas_file_post']
    readable_name = "Destrieux_aparc32k"
    local_path = subject_dir + "/" + patient_id + HCP_paths['destrieux_atlas_file_post']
    files.append({"hcp_path": hcp_path, "readable_name": readable_name, "local_path": local_path})
    
    #Desikan atlas
    hcp_path = HCP_paths['path2HCP_1200'] + patient_id + HCP_paths['atlas_file_pre'] + patient_id + HCP_paths['desikan_atlas_file_post']
    readable_name = "Desikan_aparc32k"
    local_path = subject_dir + "/" + patient_id + HCP_paths['desikan_atlas_file_post']
    files.append({"hcp_path": hcp_path, "readable_name": readable_name, "local_path": local_path})

    return files


def subject_list_HCP_1200(general_dir, save_sl):
    try:
        # Ejecutar el comando y capturar la salida
        result = subprocess.run(["aws", "s3", "ls", "s3://hcp-openaccess/HCP_1200/"], capture_output=True, text=True, check=True)

        # Decodificar la salida a un string
        output = result.stdout

        # Procesar la salida para obtener la lista de sujetos
        subject_list = []
        for line in output.splitlines():

            subject_id = line.split()[1].strip('/')
            subject_list.append(subject_id)

        subject_list = [elemento for elemento in subject_list if elemento.isdigit() and len(elemento) == 6]
        
        if save_sl:
            print(f'Saving subject list of {len(subject_list)} hcp subject')
            savemat(f"{general_dir}/hcp_1200_subject_list.mat", {'hcp1200_subject_list': subject_list})
            
        return subject_list
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando: {e}")
        return []


#Check to see if this file exists on hcp server
def check_exist_hcp(rel_subj_path, patient_id, HCP_paths):

    hcp_path = HCP_paths['path2HCP_1200'] + patient_id + rel_subj_path
    print(f'contructed path: {hcp_path}')
    print(f'\tExist? -> {rel_subj_path} || ',end='')
    args = ["aws s3 ls s3:/" + hcp_path + " " + "--human-readable"]
    completed_process = subprocess.run(args, capture_output=True, shell=True)
    out = completed_process.stdout.decode(encoding)

    #if download did not complete properly, print this and save this info
    if completed_process.returncode != 0:
        print(f'NO...\n\t{out} ')#'ERROR (likely not on server or faulty path given')
    else:
        print(f'YES...\n\t{out}')


def download_subj(subject, HCP_paths, subject_dir, patients_missing_file, r_num_all):
    
    for f in list_files(subject, HCP_paths, subject_dir, r_num_all):

        if os.path.isfile(f['local_path']):
            print(f'\t{f["readable_name"]} already exists, skipping...')
            continue

        #if file does not already exist in local directory, attempt to download it
        print(f'\tdownloading {f["readable_name"]} into {subject_dir}...',end='')
        #args = ["aws s3 cp s3:/" + f["hcp_path"] + " " + subject_dir]
        s3_path = "s3://" + f["hcp_path"].lstrip('/')
        
        try:
            completed_process = subprocess.run(["aws", "s3", "cp", s3_path, f["local_path"]], check=True, capture_output=True, text=True)
            
            #if download did not complete properly, print this and save this info
            if completed_process.returncode != 0:
                print(f'ERROR (likely not on server or faulty path given')
                patients_missing_file[f["readable_name"]].add(subject)
            else:
                print(f'success')

        except subprocess.CalledProcessError:
            print('there is a problem with this guy lol')
            patients_missing_file[f["readable_name"]].add(subject)
            
    return patients_missing_file
                    
                    
def contstruct_time_series(subject, HCP_paths, subject_dir, roi_path, timeseries_dir, r_num_all, TASK):
    
    subj_timeseries = []
    
    for f in list_files(subject, HCP_paths, subject_dir, r_num_all):
                
        file_name = f["local_path"].split('/')[-1]
                
        # only downloads the ones i'll actually be usings
        if file_name in [ 
                        f'tfMRI_{TASK}_RL_Atlas.dtseries.nii',
                        f'tfMRI_{TASK}_LR_Atlas.dtseries.nii',
                        ]:
            
            path2atlas = f'{subject_dir}/{subject}.aparc.32k_fs_LR.dlabel.nii'
            
            path2fmri = f["local_path"]
            
            name = file_name.split('_')[1]
            direction = file_name.split('_')[2]
            
            timeseries_file = f'{timeseries_dir}/timeseries_{name}_{direction}.mat'
            # try:
            time_series = get_timeseries(path2fmri, roi_path, path2atlas, subject, timeseries_file)
            subj_timeseries.append(time_series)
            
            # except Exception as e: 
            #     print(e)
            #     print(subject)
          
    return subj_timeseries


def process_files(subject, HCP_paths, subject_dir, roi_path, kind, r_num_all, corr_path, timeseries_dir):
    
    # Then find corresponding timeseries
    Path(timeseries_dir).mkdir(parents=True, exist_ok=True)
    
    for task in ['GAMBLING', 'MOTOR', 'LANGUAGE', 'EMOTION' ]:
        subj_timeseries = contstruct_time_series(subject, HCP_paths, subject_dir, roi_path, timeseries_dir = timeseries_dir, r_num_all = r_num_all, TASK = task)
        
        if kind == None:
            corr_matrix = '_'
            cov_matrix = '_'
        else:
            concatenated_ts = concatenate_timeseries(subj_timeseries)
            Path(corr_path).mkdir(parents=True, exist_ok=True)
            corr_matrix, cov_matrix = calculate_fc_matrix(concatenated_ts, kind, subject, corr_path, task)
    
    return corr_matrix, cov_matrix, subj_timeseries


def create_subject_directories(general_dir, subject, download_ts, download_fc):
    
    subject_dir = os.path.join(general_dir, 'brain_data', subject)
    timeseries_dir = os.path.join(general_dir, 'time_series', subject) if download_ts else None
    corr_dir = os.path.join(general_dir, 'corr_matrices', subject) if download_fc else None
    Path(subject_dir).mkdir(parents=True, exist_ok=True)
    
    return subject_dir, timeseries_dir, corr_dir


def evaluate_subject_directories(general_dir, subject, download_ts, download_fc, delete_files):
    timeseries_dir = os.path.join(general_dir, 'time_series', subject)
    corr_dir = os.path.join(general_dir, 'corr_matrices', subject)
    subject_dir = os.path.join(general_dir, 'brain_data', subject)

    timeseries_needed = download_ts and not os.path.exists(timeseries_dir)
    corr_needed = download_fc and not os.path.exists(corr_dir)

    # Create the subject directory if it doesn't exist or if we're not deleting files afterward
    if not os.path.exists(subject_dir) and not delete_files:
        Path(subject_dir).mkdir(parents=True, exist_ok=True)

    return subject_dir, timeseries_needed, corr_needed, timeseries_dir, corr_dir


def handle_subject_files(subject, HCP_paths, subject_dir, roi_path, patients_missing_file, r_num_all, timeseries_dir, corr_dir, kind):
    
    patients_missing_file = download_subj(subject, HCP_paths, subject_dir, patients_missing_file, r_num_all)
    
    try:
        process_files(subject, HCP_paths, subject_dir, roi_path, kind, r_num_all, corr_dir, timeseries_dir)
        
    except Exception as e:
        print(e)
        
    return patients_missing_file


#for each patient, create a local diretory in local_dir and download all files. If there is an issue downloading, print to terminal and record which patient could not download each file
def HCP_handler(r_num_all, roi_path, general_dir, kind, delete_files, download_ts, download_fc):
    
    HCP_paths = get_file_names(r_num_all)
    
    files = list_files('000', HCP_paths, subject_dir = local_dir + '000', r_num_all = r_num_all)
    
    patients_missing_file = {f["readable_name"]:set() for f in files}

    subject_list = subject_list_HCP_1200(general_dir, save_sl = True)
    
    # Here is where it downloads for each subject
    for idx, subject in enumerate(subject_list):
        
        print(f"\n\n{idx}th subject: {subject}")
        
        subject_dir, ts_needed, corr_needed, timeseries_dir, corr_dir = evaluate_subject_directories(general_dir, subject, download_ts, download_fc, delete_files)
            
        if ts_needed or corr_needed:
            
            start = time.time()
            
            #download all files into this dir
            patients_missing_file = handle_subject_files(subject, HCP_paths, subject_dir, roi_path, patients_missing_file, r_num_all, timeseries_dir, 
                                                         corr_dir, kind)
                
            if delete_files:
                shutil.rmtree(subject_dir)
            
            end = time.time()
            
            print(f"Time for download and processing: {(end-start):.1f}")

            report_missing_files(patients_missing_file)
            
    end_of_process_sumary(subject_list, patients_missing_file)


local_dir =  "/home/personal/Documents/2023_2/tesis/data_processing/data_desikan_task" # path where data will be stored
roi_path = '/home/personal/Documents/2023_2/tesis/GNN-BAA/data_processing/data/original_data/desikan_roi_zhengwu.mat'
encoding = 'utf-8'
r_num_all = [1] #[1], [2] or [1, 2]
kind = 'both' # both, covariance, correlation
delete_files = False # True or False, delete files once processed as theyre downloaded
download_ts = True # True or False, keep timeseries once calculated
download_fc = True # True or False, keep fc matrices once calculated, if kind == None it doesn't really matter

HCP_handler(r_num_all, roi_path, local_dir, kind, delete_files, download_ts, download_fc)

