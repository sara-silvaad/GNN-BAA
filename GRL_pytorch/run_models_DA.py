import subprocess
import os

# Function to run main.py with a specific model
def run_main_with_model(main_script_path, model, param):

    os.environ['PARAM'] = param
    os.environ['MODEL'] = model
    
    # Execute the main.py script
    subprocess.run(['python', main_script_path], check=True)

main_script_path = '/datos/projects/ssilva/GNNBAA/GRL_pytorch/main_DA.py'

# SUPER NODE

models = ['EncoderClassifierSCSuperNode']
intensity_super_node = ['100', '200', '500', '1000']

for model in models:
    for intensity in intensity_super_node:
        print(f"Running main.py with model={model}, intensity super node= {intensity}")
        run_main_with_model(main_script_path, model, intensity)
        print(f"Finished running main.py with model={model}, intensity super node= {intensity}")

## SPURIOUS CONNECTIONS

# models = ['EncoderClassifierSCSpurious']
# intensities = ['10']

# for model in models: 
#     for intensity in intensities:
#         print(f"Running main.py with model={model}, intensity spurious= {intensity}")
#         run_main_with_model(main_script_path, model, intensity)
#         print(f"Finished running main.py with model={model}, intensity spurious= {intensity}")

# MASK ATTRIBUTES 

# models = ['EncoderClassifierSCMaskAttributes']
# Ns = ['40'] # number of nodes to mask goes between 0 and N

# for model in models: 
#     for N in Ns:
#         print(f"Running main.py with model={model}, N = {N}")
#         run_main_with_model(main_script_path, model, N)
#         print(f"Finished running main.py with model={model}, N = {N}")

## ZONE MASK ATTRIBUTES

# models = ['EncoderClassifierSCZoneMaskAttributes']
# for model in models: 
#     print(f"Running main.py with model={model}")
#     run_main_with_model(main_script_path, model, '0')
#     print(f"Finished running main.py with model={model}")
