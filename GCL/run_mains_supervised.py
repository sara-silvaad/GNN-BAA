import subprocess
import os

# Function to run main.py with a specific model
def run_main_with_model(main_script_path_pre, tau, data_aug, norm, hidden_dim, main_script_path_ft):

    os.environ['TAU'] = tau
    os.environ['DATA_AUG'] = data_aug
    os.environ['NORM'] = norm
    os.environ['HIDDEN_DIM'] = hidden_dim

    # Execute the main.py script
    # subprocess.run(['python', main_script_path_pre], check=True)
    subprocess.run(['python', main_script_path_ft], check=True)


augmentations = ['featureMasking_edgeDropping']
taus = ['1']
norms = ['True']
hidden_dims = ['1']


main_script_path_pre = '/datos/projects/ssilva/GNNBAA/GCL/main_pre_sup.py'
main_script_path_ft = '/datos/projects/ssilva/GNNBAA/GCL/main_ft_sup.py'

for data_aug in augmentations:
    for tau in taus:
        for norm in norms:
            for hidden_dim in hidden_dims:
                print(f"Running main.py with data_aug={data_aug}, tau={tau}, norm={norm} and hidden_dim={hidden_dim}")
                run_main_with_model(main_script_path_pre, tau, data_aug, norm, hidden_dim, main_script_path_ft = main_script_path_ft)
                print(f"Finished running main.py with data_aug={data_aug}, tau={tau}, norm={norm} and hidden_dim={hidden_dim}")