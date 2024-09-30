import os
import shutil
import re
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def check_path(path_to_save, config_path='./config.py'):
        # Verificar si la ruta existe
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        exp_path = os.path.join(path_to_save, 'exp_001')
        os.makedirs(exp_path)
        shutil.copy(config_path, exp_path)
        return exp_path

    # Obtener una lista de las subcarpetas existentes que coinciden con el patrón 'exp_###'
    existing_exp = [d for d in os.listdir(path_to_save) if re.match(r'exp_\d{3}', d)]

    if not existing_exp:
        exp_path = os.path.join(path_to_save, 'exp_001')
        os.makedirs(exp_path)
        shutil.copy(config_path, exp_path)
        return exp_path

    # Encontrar el último número de experimento
    existing_exp.sort()
    last_exp = existing_exp[-1]
    last_exp_num = int(re.search(r'\d{3}', last_exp).group())
    
    # Asignar el siguiente número de experimento
    new_exp_num = last_exp_num + 1
    new_exp_str = f'exp_{new_exp_num:03d}'
    exp_path = os.path.join(path_to_save, new_exp_str)
    os.makedirs(exp_path)
    shutil.copy(config_path, exp_path)
    
    return exp_path


def save_model(path, filename, best_model):
    ckpt_name = f'{filename}.pt'
    full_path = f'{path}/{ckpt_name}'
    #save_model
    torch.save({
            'model_state_dict': best_model.state_dict(),
            }, full_path)
    
def guardar_parrafo_en_archivo(general_path):
    """
    Solicita al usuario ingresar un párrafo y lo guarda en especificaciones.txt en la ruta especificada.

    :param general_path: Ruta del directorio donde se guardará el archivo especificaciones.txt.
    """
    # Solicitar al usuario que ingrese el texto
    print("Por favor, ingrese el texto que desea guardar en especificaciones.txt (presione Enter dos veces para finalizar):")

    lineas = []
    while True:
        linea = input()
        if linea == "":
            break
        lineas.append(linea)

    # Unir las líneas en un solo string con saltos de línea
    texto = "\n".join(lineas)

    # Asegurarse de que la ruta del directorio existe
    if not os.path.exists(general_path):
        os.makedirs(general_path)

    # Guardar el texto en especificaciones.txt
    with open(os.path.join(general_path, 'especificaciones.txt'), 'w') as archivo:
        archivo.write(texto)

    print("Texto guardado en especificaciones.txt")

# Ejemplo de uso
# guardar_parrafo_en_archivo("ruta/del/directorio")
