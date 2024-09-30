import subprocess
import os

# Function to run main.py with a specific model
def run_main_with_model(main_script_path_pre, main_script_path_ft, lambda_val, data_aug):

    os.environ['LAMBDA_VAL'] = lambda_val
    os.environ['DATA_AUG'] = data_aug

    # Execute the main.py script
    subprocess.run(['python', main_script_path_pre], check=True)
    subprocess.run(['python', main_script_path_ft], check=True)


# augmentations = ['region']
# lambdas = ['0.9']

augmentations = ['featureMasking_edgeDropping']
lambdas = ['0']

main_script_path_pre = '/datos/projects/ssilva/GNNBAA/GCL/main_pre.py'
main_script_path_ft = '/datos/projects/ssilva/GNNBAA/GCL/main_ft.py'

for data_aug in augmentations:
    for lambda_val in lambdas:
        print(f"Running main.py with data_aug={data_aug}, and lambda_val={lambda_val}")
        run_main_with_model(main_script_path_pre, main_script_path_ft, lambda_val, data_aug)
        print(f"Finished running main.py with data_aug={data_aug}, and lambda_val={lambda_val}")
