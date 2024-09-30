import subprocess
import os

# Function to run main.py with a specific model
def run_main_with_model(main_script_path, model, x_type, emb_size):

    os.environ['MODEL'] = model
    os.environ['X_TYPE'] = x_type
    os.environ['EMB_SIZE'] = emb_size

    # Execute the main.py script
    subprocess.run(['python', main_script_path], check=True)


models = ['EncoderClassifierSC']

x_types = ['aligned_veps']
emb_sizes = ['2', '5', '10', '30', '60', '87']


main_script_path = '/datos/projects/ssilva/GNNBAA/GRL_pytorch/main_GCN.py'

for model in models:
    for x_type in x_types:
        for emb_size in emb_sizes:
            print(f"Running main.py with model={model}, x_type={x_type}, and emb_size={emb_size}")
            run_main_with_model(main_script_path, model, x_type, emb_size)
            print(f"Finished running main.py with model={model}, x_type={x_type}, and emb_size={emb_size}")
