import torch
import torch.nn.functional as F


def supervised_contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, data_y, tau):

    Z = F.normalize(torch.cat((graph_emb_1, graph_emb_2), dim=0), p =2, dim =1)
    y = torch.cat((data_y, data_y), dim=0)

    # los 0's
    N = y == 0 
    
    mask = torch.ones_like(Z[N]@Z[N].T) - torch.eye((Z[N]@Z[N].T).size(0)).to('cuda')
    Z_negativos = torch.exp(((Z[N]@Z[N].T)/tau)*mask)

    mask = torch.ones_like(Z[N]@Z.T) - torch.eye((Z@Z.T).size(0))[:Z[N].shape[0],:].to('cuda')
    m = torch.sum(torch.exp((Z[N]@Z.T)/tau)*mask, dim= 1)
    loss_n = (torch.log(Z_negativos/m)/N.sum()).sum()

    # los 1's
    P = y == 1 
    mask = torch.ones_like(Z[P]@Z[P].T) - torch.eye((Z[P]@Z[P].T).size(0)).to('cuda') 
    Z_positivos = torch.exp(((Z[P]@Z[P].T)/tau)*mask)
    mask = torch.ones_like(Z[P]@Z.T) - torch.eye((Z@Z.T).size(0))[:Z[P].shape[0],:].to('cuda')

    m = torch.sum(torch.exp((Z[P]@Z.T)/tau)*mask, dim= 1)

    loss_p = (torch.log(Z_positivos/m)/P.sum()).sum()

    return -(loss_n + loss_p)

def calculate_accuracy(logits, labels):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    #labels = torch.argmax(labels, dim=1)
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.shape[0]
    return accuracy.item()

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

def similarity(x, y):
    return F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim = 2)

def n_way_loss(graph_emb_1, graph_emb_2):
    batch_size = graph_emb_1.size(0)
    sim_11 = torch.exp(similarity(graph_emb_1, graph_emb_1)).to(graph_emb_1.device)
    sim_12 = torch.exp(similarity(graph_emb_1, graph_emb_2)).to(graph_emb_1.device)
    sim_22 = torch.exp(similarity(graph_emb_2, graph_emb_2)).to(graph_emb_1.device)

    mask = torch.eye(batch_size).to(graph_emb_1.device)
    denominator = (sim_11 + sim_12 - sim_11 * mask - sim_12 * mask + sim_22 - sim_22 * mask).sum(1)

    loss = -torch.log(sim_12.diag() / denominator).mean()
    return loss

def n_way_loss_with_itself(graph_emb_1, tau = 0.5):
    batch_size = graph_emb_1.size(0)
    sim = torch.exp(similarity(graph_emb_1, graph_emb_1)/tau)
    mask = torch.eye(batch_size).to(graph_emb_1.device).to(graph_emb_1.device)
    denominator = (sim - sim*mask).sum(1)
    loss = -torch.log(sim.diag()/ denominator).mean()
    return loss 

def gender_similarity_loss(graph_emb_1, data_y, tau = 0.5):
    batch_size = graph_emb_1.size(0)
    sim = torch.exp(similarity(graph_emb_1, graph_emb_1)/tau)

    same_gender = (data_y.unsqueeze(1) == data_y.unsqueeze(0)).float().to(graph_emb_1.device)
    mask = torch.eye(batch_size).to(graph_emb_1.device)
    same_gender = same_gender * (1 - mask)  # Excluir la diagonal (auto-similitud)

    numerator = (same_gender * sim).sum(1)
    denominator = sim.sum(1) - sim.diag()

    loss = -torch.log(numerator / denominator).mean()
    return loss

def loss_gender_augmented_contrastive(graph_emb, graph_emb_1, graph_emb_2, graph_emb_3, data_y, lambda_val=0.9):
    loss_gender = gender_similarity_loss(graph_emb, data_y)
    loss_n_way = n_way_loss_with_itself(graph_emb_1) + n_way_loss_with_itself(graph_emb_2) + n_way_loss_with_itself(graph_emb_3)
    return (1 - lambda_val)*loss_gender + lambda_val*loss_n_way, loss_gender, loss_n_way 
#lambda_val * n_way_loss(graph_emb_1, graph_emb_2)

def contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, data_y, lambda_val):
    loss_gender = gender_similarity_loss(graph_emb, data_y)
    loss_n_way = n_way_loss(graph_emb_1, graph_emb_2)
    return (1 - lambda_val)*loss_gender + lambda_val*loss_n_way, loss_gender, loss_n_way
