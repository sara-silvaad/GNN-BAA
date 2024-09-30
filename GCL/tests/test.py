import torch
from utils.metrics import calculate_accuracy, calculate_precision_recall_f1, loss_gender_augmented_contrastive, contrastive_loss, supervised_contrastive_loss
from data.data_augmentation import add_spurious_connections


def test_pre(model, test_loader, lambda_val, data_aug, tau, loss_recon = None):
    model.eval()
    test_loss = 0
    test_loss_gender = 0
    test_loss_n_way = 0

    with torch.no_grad():
        for data in test_loader:
            
            if data_aug == 'featureMasking_edgeDropping':
                graph_emb, graph_emb_1, graph_emb_2 = model(data)  
                if tau is not None:
                    t_loss = supervised_contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, data.y, tau = tau)   
                    t_loss_gender = torch.tensor(0.0)
                    t_loss_n_way = torch.tensor(0.0)
                else:
                    t_loss, t_loss_gender, t_loss_n_way = contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, data.y, lambda_val=lambda_val)   
            
            elif data_aug == 'region':    
                graph_emb, graph_emb_1, graph_emb_2, graph_emb_3  = model(data)  
                t_loss, t_loss_gender, t_loss_n_way = loss_gender_augmented_contrastive(graph_emb, graph_emb_1, graph_emb_2, graph_emb_3, data.y, lambda_val=lambda_val)     
            
            elif data_aug == 'featureMasking_edgeDropping_Decoder':
                reconstruction, graph_emb, graph_emb_1, graph_emb_2 = model(data)
                t_reconstruction_loss = loss_recon(reconstruction, data.fc.to('cuda'))
                
                if tau is not None:  # Better to use 'is not None' when checking for None
                    t_loss = supervised_contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, data.y, tau=tau)                    
                    t_loss_gender = torch.tensor(0.0)
                    t_loss_n_way = torch.tensor(0.0)
                else:
                	t_loss, t_loss_gender, t_loss_n_way = contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, data.y, lambda_val=lambda_val)
            
            if loss_recon is not None:
                 t_loss = t_loss + 0.35 * t_reconstruction_loss     
                 
            test_loss += t_loss.item()
            test_loss_gender += t_loss_gender.item()
            test_loss_n_way += t_loss_n_way.item()
            
    # Calculating averages
    test_loss_avg = test_loss / len(test_loader)
    test_loss_gender_avg = test_loss_gender / len(test_loader)
    test_loss_n_way_avg = test_loss_n_way / len(test_loader)

    # Compiling results into a dictionary
    results = {
        'test_loss': test_loss_avg,
        'test_loss_gender': test_loss_gender_avg,
        'test_loss_n_way': test_loss_n_way_avg
    }

    print(f'Test Loss: {test_loss_avg:.2f}, Test Loss Gender: {test_loss_gender_avg:.2f}, Test Loss N-way: {test_loss_n_way_avg:.2f}')
    return results

def test_ft(encoder, classifier, test_loader, criterion_classif, spurious_connetions = None):
    encoder.eval()
    classifier.eval()
    test_loss = 0
    test_accuracy = 0
    test_precision = 0
    test_recall = 0
    test_f1 = 0

    with torch.no_grad():
        for data in test_loader:
            # Obtener las embeddings de los nodos del encoder
            x, edge_index_sc, edge_weight_sc, batch = data.x, data.edge_index_sc, data.edge_weight_sc, data.batch
            
            if spurious_connetions is not None:
                edge_index_sc, edge_weight_sc = add_spurious_connections(data.sc, spurious_connetions, 87)
                    
            node_emb = encoder.encoder(x, edge_index_sc, edge_weight_sc)
            graph_emb = encoder.graph_embedding(node_emb, batch)
            
            # Obtener las predicciones del clasificador
            predictions = torch.squeeze(classifier(graph_emb)).float()
            
            # Calcular la pérdida
            t_loss = criterion_classif(predictions, data.y.long())
            test_loss += t_loss.item()

            # Calcular métricas
            acc = calculate_accuracy(predictions, data.y)
            test_accuracy += acc

            precision, recall, f1_score = calculate_precision_recall_f1(predictions, data.y)
            test_precision += precision
            test_recall += recall
            test_f1 += f1_score

    # Calcular promedios
    test_loss_avg = test_loss / len(test_loader)
    test_accuracy_avg = test_accuracy / len(test_loader)
    test_precision_avg = test_precision / len(test_loader)
    test_recall_avg = test_recall / len(test_loader)
    test_f1_avg = test_f1 / len(test_loader)

    # Compilar resultados en un diccionario
    results = {
        'test_loss': test_loss_avg,
        'test_accuracy': test_accuracy_avg,
        'test_precision': test_precision_avg,
        'test_recall': test_recall_avg,
        'test_f1': test_f1_avg
    }

    print(f'Test Loss: {test_loss_avg:.2f}, Test Acc: {test_accuracy_avg:.2f}, '
          f'Test Precision: {test_precision_avg:.2f}, Test Recall: {test_recall_avg:.2f}, '
          f'Test F1: {test_f1_avg:.2f}')
    return results
