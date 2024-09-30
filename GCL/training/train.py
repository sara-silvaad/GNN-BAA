from utils.metrics import calculate_precision_recall_f1, calculate_accuracy, loss_gender_augmented_contrastive, contrastive_loss, supervised_contrastive_loss
import torch
import os
from utils.set_optimizer import set_optimizer
from data.data_augmentation import add_spurious_connections

def train_and_evaluate_pre(model, train_loader, val_loader, epochs, optimizer, early_stopping_patience, lambda_val, data_aug, tau, loss_recon = None):
    
    results = {
        'train_loss': [],
        'train_loss_gender': [],
        'train_loss_n_way': [],
        'val_loss': [],
        'val_loss_gender': [],
        'val_loss_n_way': []
    }

    # Inicialización para Early Stopping
    best_model = {}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):  # Training and Validation loop
        model.train()
        total_loss = 0
        total_loss_gender = 0
        total_loss_n_way = 0

        for data in train_loader:
            
            y = data.y.to('cuda')
            optimizer.zero_grad()

            if data_aug == 'featureMasking_edgeDropping':
                graph_emb, graph_emb_1, graph_emb_2 = model(data)
                if tau is not None:  # Better to use 'is not None' when checking for None
                    loss = supervised_contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, y, tau=tau)
                    loss_gender = torch.tensor(0.0)
                    loss_n_way = torch.tensor(0.0)
                else:
                	loss, loss_gender, loss_n_way = contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, y, lambda_val=lambda_val) 
                
            elif data_aug == 'region':
                graph_emb, graph_emb_1, graph_emb_2, graph_emb_3  = model(data) 
                loss, loss_gender, loss_n_way = loss_gender_augmented_contrastive(graph_emb, graph_emb_1, graph_emb_2, graph_emb_3, y, lambda_val=lambda_val)
            
            elif data_aug == 'featureMasking_edgeDropping_Decoder':
                reconstruction, graph_emb, graph_emb_1, graph_emb_2 = model(data)
                reconstruction_loss = loss_recon(reconstruction, data.fc.to('cuda'))
                
                if tau is not None:  # Better to use 'is not None' when checking for None
                    loss = supervised_contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, y, tau=tau)                    
                    loss_gender = torch.tensor(0.0)
                    loss_n_way = torch.tensor(0.0)
                else:
                	loss, loss_gender, loss_n_way = contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, y, lambda_val=lambda_val)
            
            if loss_recon is not None:
                loss = loss + 0.35 * reconstruction_loss
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_gender += loss_gender.item()
            total_loss_n_way += loss_n_way.item()

        train_loss_avg = total_loss / len(train_loader)
        train_loss_gender = total_loss_gender / len(train_loader)
        train_loss_n_way = total_loss_n_way / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_loss_gender = 0
        val_loss_n_way = 0
        with torch.no_grad():
            for data in val_loader:
                y = data.y.to('cuda')
                if data_aug == 'featureMasking_edgeDropping':
                    graph_emb, graph_emb_1, graph_emb_2 = model(data)  
                    if tau is not None:  # Better to use 'is not None' when checking for None
                        v_loss = supervised_contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, y, tau=tau)
                        v_loss_gender = torch.tensor(0.0)
                        v_loss_n_way = torch.tensor(0.0)
                    else:
                        v_loss, v_loss_gender, v_loss_n_way = contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, y, lambda_val=lambda_val)   
                elif data_aug == 'region':    
                    graph_emb, graph_emb_1, graph_emb_2, graph_emb_3 = model(data) 
                    v_loss, v_loss_gender, v_loss_n_way = loss_gender_augmented_contrastive(graph_emb, graph_emb_1, graph_emb_2, graph_emb_3, y, lambda_val=lambda_val)                 
                
                elif data_aug == 'featureMasking_edgeDropping_Decoder':
                    reconstruction, graph_emb, graph_emb_1, graph_emb_2 = model(data)
                    v_reconstruction_loss = loss_recon(reconstruction, data.fc.to('cuda'))
                    
                    if tau is not None:  # Better to use 'is not None' when checking for None
                        v_loss = supervised_contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, y, tau=tau)                    
                        v_loss_gender = torch.tensor(0.0)
                        v_loss_n_way = torch.tensor(0.0)
                    else:
                        v_loss, v_loss_gender, v_loss_n_way = contrastive_loss(graph_emb, graph_emb_1, graph_emb_2, y, lambda_val=lambda_val)

                if loss_recon is not None:
                    v_loss = v_loss + 0.35 * v_reconstruction_loss
                    
                val_loss += v_loss.item()
                val_loss_gender += v_loss_gender.item()
                val_loss_n_way += v_loss_n_way.item()


        val_loss_avg = val_loss / len(val_loader)
        val_loss_gender_avg = val_loss_gender / len(val_loader)
        val_loss_n_way_avg = val_loss_n_way / len(val_loader)

        # Append epoch results to the results dict
        results['train_loss'].append(train_loss_avg)
        results['train_loss_gender'].append(train_loss_gender)
        results['train_loss_n_way'].append(train_loss_n_way)
        
        results['val_loss'].append(val_loss_avg)
        results['val_loss_gender'].append(val_loss_gender_avg)
        results['val_loss_n_way'].append(val_loss_n_way_avg)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss_avg:.2f}, Train Loss Gender: {train_loss_gender:.2f}, Train Loss N-way: {train_loss_n_way:.2f}, Val Loss: {val_loss_avg:.2f}, Val Loss Gender: {val_loss_gender_avg:.2f}, Val Loss N-way: {val_loss_n_way_avg:.2f}')

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
            best_model['model'] = model
            best_model['epoch'] = epoch
            best_model['val_loss'] = val_loss_avg
            best_model['optimizer'] = optimizer
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
    return results, best_model

def train_and_evaluate_ft(encoder, classifier, train_loader, val_loader, epochs, lr, criterion, early_stopping_patience, epochs_all, spurious_connetions = None):
    results = {
        'train_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
    }
    best_model = {}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    optimizer = set_optimizer(encoder= encoder, classifier= classifier, train_encoder= False, train_classifier= True, lr= lr)

    for epoch in range(epochs):
        if epoch == epochs_all:
            optimizer = set_optimizer(encoder= encoder, classifier= classifier, train_encoder= True, train_classifier= True, lr= lr*0.1)
        
        if epochs < epochs_all:
            encoder.eval()  # Cambia a modo de entrenamiento
        else:
            encoder.train()
                
        classifier.train()
        
        total_loss = 0
        train_accuracy = 0
        train_precision = 0
        train_recall = 0
        train_f1 = 0

        for data in train_loader:
            optimizer.zero_grad()
            
            x, edge_index_sc, edge_weight_sc, batch = data.x, data.edge_index_sc, data.edge_weight_sc, data.batch
            
            if spurious_connetions is not None:
                edge_index_sc, edge_weight_sc = add_spurious_connections(data.sc, spurious_connetions, 87)
                
            node_emb = encoder.encoder(x, edge_index_sc, edge_weight_sc)
            graph_emb = encoder.graph_embedding(node_emb, batch)
            predictions = torch.squeeze(classifier(graph_emb)).float()
            loss = criterion(predictions, data.y.long())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            train_accuracy += calculate_accuracy(predictions, data.y.long())
            macro_precision, macro_recall, macro_f1_score = calculate_precision_recall_f1(predictions, data.y)
            train_precision += macro_precision
            train_recall += macro_recall
            train_f1 += macro_f1_score

        train_loss_avg = total_loss / len(train_loader)
        train_accuracy_avg = train_accuracy / len(train_loader)
        train_f1_avg = train_f1 / len(train_loader)
        train_precision_avg = train_precision / len(train_loader)
        train_recall_avg = train_recall / len(train_loader)

        classifier.eval()
        encoder.eval()

        val_loss = 0
        val_accuracy = 0
        val_precision = 0
        val_recall = 0
        val_f1 = 0

        with torch.no_grad():
            for data in val_loader:
                
                x, edge_index_sc, edge_weight_sc, batch = data.x, data.edge_index_sc, data.edge_weight_sc, data.batch
            
                if spurious_connetions is not None:
                    edge_index_sc, edge_weight_sc = add_spurious_connections(data.sc, spurious_connetions, 87)
                
                node_emb = encoder.encoder(x, edge_index_sc, edge_weight_sc)
                graph_emb = encoder.graph_embedding(node_emb, batch)
                predictions = torch.squeeze(classifier(graph_emb)).float()
                v_loss = criterion(predictions, data.y.long())
                val_loss += v_loss.item()

                val_accuracy += calculate_accuracy(predictions, data.y)
                macro_precision, macro_recall, macro_f1_score = calculate_precision_recall_f1(predictions, data.y)
                val_precision += macro_precision
                val_recall += macro_recall
                val_f1 += macro_f1_score

        val_loss_avg = val_loss / len(val_loader)
        val_accuracy_avg = val_accuracy / len(val_loader)
        val_f1_avg = val_f1 / len(val_loader)
        val_precision_avg = val_precision / len(val_loader)
        val_recall_avg = val_recall / len(val_loader)

        # Append epoch results to the results dict
        results['train_loss'].append(train_loss_avg)
        results['train_accuracy'].append(train_accuracy_avg)
        results['train_precision'].append(train_precision_avg)
        results['train_recall'].append(train_recall_avg)
        results['train_f1'].append(train_f1_avg)
        results['val_loss'].append(val_loss_avg)
        results['val_accuracy'].append(val_accuracy_avg)
        results['val_precision'].append(val_precision_avg)
        results['val_recall'].append(val_recall_avg)
        results['val_f1'].append(val_f1_avg)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss_avg:.2f}, Train Acc: {train_accuracy_avg:.2f}, Train Precision: {train_precision_avg:.2f}, Train Recall: {train_recall_avg:.2f}, Train F1: {train_f1_avg:.2f}, Val Loss: {val_loss_avg:.2f}, Val Acc: {val_accuracy_avg:.2f}, Val Precision: {val_precision_avg:.2f}, Val Recall: {val_recall_avg:.2f}, Val F1: {val_f1_avg:.2f}')

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
            best_model['model'] = classifier
            best_model['epoch'] = epoch
            best_model['val_accuracy'] = val_accuracy_avg
            best_model['val_loss'] = val_loss_avg
            best_model['optimizer'] = optimizer
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break

    return results, best_model
