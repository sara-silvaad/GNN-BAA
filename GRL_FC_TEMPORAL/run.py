import subprocess
import os

# Function to run main.py with a specific model
def run_main_with_model(main_script_path, model, param):

    os.environ['PARAM'] = param
    os.environ['MODEL'] = model
    
    # Execute the main.py script
    subprocess.run(['python', main_script_path], check=True)

main_script_path = '/datos/projects/ssilva/GNNBAA/GRL_FC_TEMPORAL/main.py'


# # FC-FC ENC-DEC

# models = ['EncoderDecoderFCFCTemporal']
# taus = ['0.2']
# for model in models: 
#     for tau in taus:
#         print(f"Running main.py with model={model}, tau = {tau}")
#         run_main_with_model(main_script_path, model, tau)
#         print(f"Finished running main.py with model={model}, tau = {tau}")
        

# FC RANDOM TEMPORAL

models = ['EncoderClassifierFCRandomTemporal']
for model in models: 
    print(f"Running main.py with model={model}")
    run_main_with_model(main_script_path, model, '0')
    print(f"Finished running main.py with model={model}")