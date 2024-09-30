import pandas as pd
import matplotlib.pyplot as plt
import os

def save_average_results(results, test_results, filename, path):
    # Para results:
    # Inicializa un diccionario para guardar los promedios de la última época
    promedios_ultima_epoca = {}

    # Itera a través de las claves (métricas) en el primer elemento de la lista para establecer las métricas a calcular
    for clave in results[0].keys():
        # Calcula el promedio de la última época para cada métrica
        suma = sum(result[clave][-1] for result in results)
        promedio = suma / len(results)
        promedios_ultima_epoca[clave] = promedio

    df = pd.DataFrame([promedios_ultima_epoca])
    os.makedirs(f'{path}/promedios/train_and_val', exist_ok = True)
    df.to_excel(f'{path}/promedios/train_and_val/{filename}.xlsx', index=False)

    # Para test_results:
    averages = {key: sum(d[key] for d in test_results) / len(test_results) for key in test_results[0]}
    df = pd.DataFrame([averages])
    os.makedirs(f'{path}/promedios/test', exist_ok = True)
    df.to_excel(f'{path}/promedios/test/{filename}.xlsx', index=False)

def save_to_excel_train(data, filename, path):
    """
    Save data to an Excel file.

    Parameters:
    - data (dict): A dictionary where keys are column names and values are lists of numbers.
    - filename (str): The path and name of the Excel file to save the data to.
    """
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    # Save the DataFrame to an Excel file
    os.makedirs(f'{path}/train', exist_ok = True)
    df.to_excel(f'{path}/train/{filename}.xlsx', index=False)
    return filename

def save_to_excel_test(data, filename, path):
    """
    Save data to an Excel file.

    Parameters:
    - data (dict): A dictionary where keys are column names and values are lists of numbers.
    - filename (str): The path and name of the Excel file to save the data to.
    """
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([data])
    # Save the DataFrame to an Excel file
    os.makedirs(f'{path}/test', exist_ok = True)
    df.to_excel(f'{path}/test/{filename}.xlsx', index=False)
    return filename

import matplotlib.pyplot as plt

def create_and_save_plots(data, filename, path):
    """
    Create a plot for each specified pair of metrics (train vs validation) if they exist in data
    and save them as a single image.

    Parameters:
    - data (dict): A dictionary where keys are metric names and values are lists of metric values per epoch.
    - filename (str): The path and name of the image file to save the plots to.
    """
    metrics_pairs = [
        ('train_loss', 'val_loss'),
        ('train_accuracy', 'val_accuracy'),
        ('train_f1', 'val_f1'),
        ('train_mse', 'val_mse')  # Including MSE metrics conditionally
    ]

    # Filter out metric pairs that are not fully present in data
    valid_metrics_pairs = [(train, val) for train, val in metrics_pairs if train in data and val in data]

    # Determine the number of rows needed based on the number of valid metric pairs
    n_rows = len(valid_metrics_pairs)
    n_cols = 1  # Columns are fixed at 1 since we're plotting train vs validation side by side

    # Create a figure to contain the plots with dynamic size
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 5))
    fig.suptitle('Training and Validation Metrics Comparison', fontsize=16)

    if n_rows > 1:
        axs_flat = axs.flat
    else:
        axs_flat = [axs] if n_rows == 1 else [fig.add_subplot(1,1,1)]

    # Plot each pair of valid metrics in its respective subplot
    for ax, (train_key, val_key) in zip(axs_flat, valid_metrics_pairs):
        ax.plot(data[train_key], label=f'Train {train_key.split("_")[1]}')
        ax.plot(data[val_key], label=f'Validation {val_key.split("_")[1]}')
        ax.set_title(f'{train_key.split("_")[1]} Comparison')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    os.makedirs(f'{path}/plots', exist_ok = True)
    plt.savefig(f'{path}/plots/{filename}.png')
    plt.close()  # Close the plot to save resources

# Reemplaza 'data' con tu diccionario de datos actual y 'filename' con la ruta y nombre de archivo deseado
# create_and_save_plots(data, 'metrics_comparison.png')


def create_plots_and_save_results(results, results_test, filename, path):
    save_to_excel_train(results,filename, path)
    create_and_save_plots(results,filename, path)
    save_to_excel_test(results_test, filename, path)


##########################
import numpy as np

human_labs = ["L", "R", "L", "R", "BS", "L", "R", "L", "R", "L", "R", "L", "R", "L", "R", "L", "R", "L", "R",  
              "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L",
              "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R",
                "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R"]


coordinates = [
    # [-9, 9, -8], # empieza subcortical
    # [9, 9, -8],
    # [-22, -3, -18],
    # [22, -3, -18],
    # [0, -33, -23], # brain stem
    # [-13, 15, 9],
    # [13, 15, 9],
    # [-17, -56, -36],
    # [17, -56, -36],
    # [-12, -15, -8],
    # [12, -15, -8],
    # [-22, -25, -9],
    # [22, -25, -9],
    # [-21, 4, 0],
    # [21, 4, 0],
    # [-25, 8, 3],
    # [25, 8, 3],
    # [-12, -15, 8],
    # [12, -15, 8], # termina subcortical
    [-52.7, -44.5, 4.6],  # lBKS
    [-6.6, 18, 26.1],     # lCAC
    [-34.6, 10.2, 42.8],  # lCMF
    [-8.7, -79.6, 18],    # lCUN
    [-25.8, -7.6, -31.6], # lENT
    [-35.7, -43.3, -19.7],# lFUS
    [-40, -66.4, 27.3],   # lIP
    [-48.9, -34.4, -22.2],# lIT
    [-9.8, -44.8, 16.9],  # lIST
    [-29.7, -86.9, -1],   # lLO
    [-24, 28.6, -14.4],   # lLOF
    [-16.5, -66.8, -4.3], # lLIN
    [-8, 34.9, -14.9],    # lMOF
    [-55.6, -31.1, -12.9],# lMT
    [-24.7, -31.2, -17.4],# lPHIP
    [-10, -28.7, 56.1],   # lPARA
    [-44.6, 14.6, 13.1],  # lPOP
    [-41, 38.8, -11.1],   # lPOB
    [-42.4, 30.6, 2.3],   # lPT
    [-13.9, -80.6, 6],    # lPERI
    [-42.3, -23.8, 43.6], # lPOC
    [-7.3, -17.4, 35.7],  # lPCG
    [-37.8, -10.7, 42.1], # lPRC
    [-11.6, -57.5, 36.7], # lPREC
    [-6.8, 33.9, 1.6],    # lRAC
    [-31.3, 41.2, 16.5],  # lRMF
    [-12.6, 22.9, 42.4],  # lSF
    [-22.8, -60.9, 46.3], # lSP
    [-52.1, -17.8, -4.4], # lST
    [-50.4, -38.8, 31],   # lSUPRA
    [-8.6, 61.7, -8.7],   # lFP
    [-32.8, 8.4, -34.8],  # lTP
    [-44, -24.2, 6],      # lTRANS
    [-34.2, -4.3, 2.2],   # lINS
    [51.9, -40.6, 5.6],   # rBKS
    [7.3, 18.7, 26.3],    # rCAC
    [34.9, 11.8, 43],     # rCMF
    [8.7, -80.1, 19],     # rCUN
    [26.2, -6.8, -31.9],  # rENT
    [35.9, -43, -19.2],   # rFUS
    [42.8, -60.9, 28.1],  # rIP
    [49.3, -31.7, -23],   # rIT
    [8.9, -45.4, 17.6],   # rIST
    [30.3, -86.3, 0.5],   # rLO
    [23.6, 28.5, -15.2],  # rLOF
    [16.8, -66.3, -3.6],  # rLIN
    [8.8, 35.7, -14.8],   # rMOF
    [55.9, -29.5, -12.9], # rMT
    [26.1, -31.3, -16.2], # rPHIP
    [9.9, -27.4, 55.6],   # rPARAC
    [44.9, 14.4, 14.2],   # rPOP
    [42.1, 39.2, -10],    # rPOB
    [45, 29.7, 4.5],      # rPT
    [14, -79.7, 6.7],     # rPERI
    [41.6, -22.4, 43.8],  # rPOC
    [7.6, -17.1, 36.2],   # rPCG
    [36.8, -9.9, 43.5],   # rPRC
    [11.7, -56.5, 37.7],  # rPREC
    [8, 33.5, 2.1],       # rRAC
    [32.3, 40.9, 17.3],   # rRMF
    [13.4, 24.7, 42],     # rSF
    [22.6, -59.5, 48.1],  # rSP
    [53, -14, -5.5],      # rST
    [50.6, -33.3, 30.7],  # rSUPRA
    [10.3, 61.1, -10],    # rFP
    [34, 8.4, -33.1],     # rTP
    [44.8, -22.4, 6.5],   # rTRANS
    [35.1, -3.9, 2.4]     # rINS
    ]

node_coords = np.array(coordinates)

from nilearn import plotting
from matplotlib.pyplot import cm
import matplotlib as mpl

def generate_embedding_figures(x):
    
    node_vmin = np.min(x) - 0.01
    node_vmax = np.max(x) + 0.01
    # node_vmin = 0
    # node_vmax = 17
    display = plotting.plot_markers(x, node_coords, node_size='auto', node_cmap=mpl.colormaps['viridis'], node_vmin=node_vmin, node_vmax=node_vmax)

# Definir las regiones de interés
# men
# highlighted_coords = np.array([
#     [-40, -66.4, 27.3],  # lIP
#     [42.8, -60.9, 28.1],  # rIP
#     [55.9, -29.5, -12.9],  # rMT
#     [22.6, -59.5, 48.1],  # rSP
#     [8.7, -80.1, 19],  # rCUN
#     [-11.6, -57.5, 36.7],  # lPREC
#     [11.7, -56.5, 37.7],  # rPREC
#     [-37.8, -10.7, 42.1],  # lPRC
#     [36.8, -9.9, 43.5]   # rPRC
# ])

#women
# highlighted_coords = [
#     [-34.6, 10.2, 42.8],  # lCMF
#     [-6.6, 18, 26.1],     # lCAC
#     [-44.6, 14.6, 13.1],  # lPOP
#     [-24, 28.6, -14.4],   # lLOF
#     [-34.2, -4.3, 2.2],   # lINS
#     [34.9, 11.8, 43],     # rCMF
#     [7.3, 18.7, 26.3],    # rCAC
#     [44.9, 14.4, 14.2],   # rPOP
#     [23.6, 28.5, -15.2],  # rLOF
#     [35.1, -3.9, 2.4],    # rINS
#     [-24.7, -31.2, -17.4],# lPHIP
#     [26.1, -31.3, -16.2]  # rPHIP
# ]

highlighted_coords = [
    [-11.6, -57.5, 36.7], # lPREC
    [-31.3, 41.2, 16.5],  # lRMF
    [11.7, -56.5, 37.7],  # rPREC
    [32.3, 40.9, 17.3],   # rRMF
]

coordinates = [
    [-9, 9, -8], # empieza subcortical
    [9, 9, -8],
    [-22, -3, -18],
    [22, -3, -18],
    [0, -33, -23], # brain stem
    [-13, 15, 9],
    [13, 15, 9],
    [-17, -56, -36],
    [17, -56, -36],
    [-12, -15, -8],
    [12, -15, -8],
    [-22, -25, -9],
    [22, -25, -9],
    [-21, 4, 0],
    [21, 4, 0],
    [-25, 8, 3],
    [25, 8, 3],
    [-12, -15, 8],
    [12, -15, 8], # termina subcortical
    [-52.7, -44.5, 4.6],  # lBKS # arranca hemisferio izquierdo
    [-6.6, 18, 26.1],     # lCAC
    [-34.6, 10.2, 42.8],  # lCMF
    [-8.7, -79.6, 18],    # lCUN
    [-25.8, -7.6, -31.6], # lENT
    [-35.7, -43.3, -19.7],# lFUS
    [-40, -66.4, 27.3],   # lIP
    [-48.9, -34.4, -22.2],# lIT
    [-9.8, -44.8, 16.9],  # lIST
    [-29.7, -86.9, -1],   # lLO
    [-24, 28.6, -14.4],   # lLOF
    [-16.5, -66.8, -4.3], # lLIN
    [-8, 34.9, -14.9],    # lMOF
    [-55.6, -31.1, -12.9],# lMT
    [-24.7, -31.2, -17.4],# lPHIP
    [-10, -28.7, 56.1],   # lPARA
    [-44.6, 14.6, 13.1],  # lPOP
    [-41, 38.8, -11.1],   # lPOB
    [-42.4, 30.6, 2.3],   # lPT
    [-13.9, -80.6, 6],    # lPERI
    [-42.3, -23.8, 43.6], # lPOC
    [-7.3, -17.4, 35.7],  # lPCG
    [-37.8, -10.7, 42.1], # lPRC
    [-11.6, -57.5, 36.7], # lPREC
    [-6.8, 33.9, 1.6],    # lRAC
    [-31.3, 41.2, 16.5],  # lRMF
    [-12.6, 22.9, 42.4],  # lSF
    [-22.8, -60.9, 46.3], # lSP
    [-52.1, -17.8, -4.4], # lST
    [-50.4, -38.8, 31],   # lSUPRA
    [-8.6, 61.7, -8.7],   # lFP
    [-32.8, 8.4, -34.8],  # lTP
    [-44, -24.2, 6],      # lTRANS
    [-34.2, -4.3, 2.2],   # lINS
    [51.9, -40.6, 5.6],   # rBKS # arranca hemisferio derecho
    [7.3, 18.7, 26.3],    # rCAC
    [34.9, 11.8, 43],     # rCMF
    [8.7, -80.1, 19],     # rCUN
    [26.2, -6.8, -31.9],  # rENT
    [35.9, -43, -19.2],   # rFUS
    [42.8, -60.9, 28.1],  # rIP
    [49.3, -31.7, -23],   # rIT
    [8.9, -45.4, 17.6],   # rIST
    [30.3, -86.3, 0.5],   # rLO
    [23.6, 28.5, -15.2],  # rLOF
    [16.8, -66.3, -3.6],  # rLIN
    [8.8, 35.7, -14.8],   # rMOF
    [55.9, -29.5, -12.9], # rMT
    [26.1, -31.3, -16.2], # rPHIP
    [9.9, -27.4, 55.6],   # rPARAC
    [44.9, 14.4, 14.2],   # rPOP
    [42.1, 39.2, -10],    # rPOB
    [45, 29.7, 4.5],      # rPT
    [14, -79.7, 6.7],     # rPERI
    [41.6, -22.4, 43.8],  # rPOC
    [7.6, -17.1, 36.2],   # rPCG
    [36.8, -9.9, 43.5],   # rPRC
    [11.7, -56.5, 37.7],  # rPREC
    [8, 33.5, 2.1],       # rRAC
    [32.3, 40.9, 17.3],   # rRMF
    [13.4, 24.7, 42],     # rSF
    [22.6, -59.5, 48.1],  # rSP
    [53, -14, -5.5],      # rST
    [50.6, -33.3, 30.7],  # rSUPRA
    [10.3, 61.1, -10],    # rFP
    [34, 8.4, -33.1],     # rTP
    [44.8, -22.4, 6.5],   # rTRANS
    [35.1, -3.9, 2.4]     # rINS
    ]


def generate_embedding_figures_highlited(x):
    node_vmin = np.min(x) - 0.01
    node_vmax = np.max(x) + 0.01

    display = plotting.plot_markers(
        x, 
        node_coords, 
        node_size='auto', 
        node_cmap=plt.cm.viridis, 
        node_vmin=node_vmin, 
        node_vmax=node_vmax
    )
    
    # Agregar un contorno alrededor de las regiones destacadas
    for coord in highlighted_coords:
        display.add_markers(
            marker_coords=[coord],
            marker_color='red',
            marker_size=100,  # Ajustar el tamaño del contorno según se necesite
            alpha=0.7
        )

    return display