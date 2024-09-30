import torch
import torch.nn.functional as F
from config import NUM_CLASSES 

######################### Cross entropy #####################################################

# def calculate_accuracy(logits, labels):
#     probs = torch.sigmoid(logits)
#     # Divide el rango [0, 1] en intervalos igualmente espaciados
#     thresholds = torch.linspace(0, 2, steps=NUM_CLASSES + 1)
#     # Asigna cada probabilidad a un intervalo, desplazando por -1 para ajustar índices correctamente
#     preds = torch.bucketize(probs, boundaries=thresholds[1:-1])  # Se omiten los extremos para usarlos implícitamente como bordes

#     # Compara las predicciones con las etiquetas verdaderas
#     correct = (preds == labels).float().sum()
#     # Calcula la precisión
#     accuracy = correct / labels.size(0)
#     return accuracy.item()

# def calculate_precision_recall_f1(logits, labels):
#     # Calcula las probabilidades usando sigmoid
#     probs = torch.sigmoid(logits)
#     # Divide el rango [0, 1] en intervalos igualmente espaciados
#     thresholds = torch.linspace(0, 1, steps=NUM_CLASSES + 1)
#     # Asigna cada probabilidad a un intervalo
#     preds = torch.bucketize(probs, boundaries=thresholds) - 1

#     # Verdaderos positivos (TP), falsos positivos (FP), y falsos negativos (FN)
#     TP = ((preds == 1) & (labels == 1)).float().sum()
#     FP = ((preds == 1) & (labels == 0)).float().sum()
#     FN = ((preds == 0) & (labels == 1)).float().sum()

#     # Precisión, recall y F1-score
#     precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0.0)
#     recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0.0)
#     f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else torch.tensor(0.0)

#     return precision.item(), recall.item(), f1_score.item()

######################### FOR MULTIPLE CLASSES #########################################

def calculate_accuracy(logits, labels):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    #labels = torch.argmax(labels, dim=1)
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.shape[0]
    return accuracy.item()

# def calculate_accuracy(tensor_list, labels):
#     # Concatenar las probabilidades de los logits a lo largo de una nueva dimensión
#     all_probs = torch.stack([torch.softmax(logits, dim=1) for logits in tensor_list], dim=0)

#     # Obtener las predicciones para cada tensor (máximo en la dimensión de las clases)
#     all_preds = torch.argmax(all_probs, dim=2)

#     # Realizar la votación mayoritaria. Si el número de tensores es par, puede resultar en un empate,
#     # aquí podrías agregar una regla para desempatar o simplemente tomar el primer máximo.
#     preds, indices = torch.mode(all_preds, dim=0)

#     # Calcula la cantidad de predicciones correctas
#     correct = (preds == labels).float().sum()

#     # Calcular la exactitud
#     accuracy = correct / labels.shape[0]

#     return accuracy.item()

def calculate_precision_recall_f1(logits, labels):
    # Convertir logits a probabilidades y luego a predicciones binarias
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    # labels = torch.argmax(labels, dim=1)

    # Verdaderos positivos (TP), falsos positivos (FP), y falsos negativos (FN)
    TP = ((preds == 1) & (labels == 1)).float().sum()
    FP = ((preds == 1) & (labels == 0)).float().sum()
    FN = ((preds == 0) & (labels == 1)).float().sum()

    # Precisión, recall y F1-score
    precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0.0)
    recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0.0)
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else torch.tensor(0.0)

    return precision.item(), recall.item(), f1_score.item()

############################# FOR 2 CLASSES ######################################

# def calculate_accuracy(logits, labels):
#     probs = torch.sigmoid(logits)  # Calcula la probabilidad
#     preds = (probs >= 0.5).float()  # Convierte las probabilidades a predicciones binarias
#     correct = (preds == labels).float().sum()  # Cuenta las predicciones correctas
#     accuracy = correct / labels.shape[0]  # Calcula la precisión
#     return accuracy.item()

# def calculate_precision_recall_f1(logits, labels):
#     probs = torch.sigmoid(logits)  # Convierte logits a probabilidades
#     preds = (probs >= 0.5).float()  # Convierte probabilidades a predicciones binarias

#     # Verdaderos positivos (TP), falsos positivos (FP), y falsos negativos (FN)
#     TP = ((preds == 1) & (labels == 1)).float().sum()
#     FP = ((preds == 1) & (labels == 0)).float().sum()
#     FN = ((preds == 0) & (labels == 1)).float().sum()

#     # Precisión, recall y F1-score
#     precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0.0)
#     recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0.0)
#     f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else torch.tensor(0.0)

#     return precision.item(), recall.item(), f1_score.item()
