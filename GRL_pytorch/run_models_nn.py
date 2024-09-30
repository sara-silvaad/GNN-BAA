import subprocess
import os


def run_main_with_model(main_script_path,  model, neg, num_node):

    os.environ['MODEL'] = model
    os.environ['NEGS'] = neg
    os.environ['NUM_NODES'] = num_node

    # Execute the main.py script
    subprocess.run(['python', main_script_path], check=True)
    
main_script_path = '/home/personal/Documents/2023_2/tesis/GNN-BAA/GRL_pytorch/main_nn.py'

models = ['FullyConnectedFC', 'FullyConnectedSCFC']

num_nodes = ['87', '68']

negs = ['False', 'True']

for model in models:
    for neg in negs:
        for num_node in num_nodes:
            print(f"Running main.py with MODEL={model}, negs={neg}, num_nodes={num_node}")
            run_main_with_model( main_script_path, model, neg, num_node)
            print(f"Finished running main.py with MODEL={model}, negs={neg}, num_nodes={num_node}")