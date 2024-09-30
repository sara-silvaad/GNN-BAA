import torch
from utils.metrics import calculate_accuracy, calculate_precision_recall_f1
import os
import random
from utils.utils import save_model

def train_and_evaluate_classifier(model,path, filename, train_loader, val_loader, epochs, optimizer, criterion_classif, early_stopping_patience, model_path, save_model_):
    ckpt_path = f'{path}/ckpt/'
    os.makedirs(ckpt_path, exist_ok=True)

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

    # Inicialización para Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):  # Training and Validation loop
        model.train()
        total_loss = 0
        train_accuracy = 0
        train_precision = 0
        train_recall = 0
        train_f1 = 0

        for data in train_loader:
            
            y = data.y.long().to('cuda:1')
            
            optimizer.zero_grad()

            classif_logits = model(data)
            
            loss = criterion_classif(classif_logits, y) 

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            train_accuracy += calculate_accuracy(classif_logits, y)
            macro_precision, macro_recall, macro_f1_score = calculate_precision_recall_f1(classif_logits, y)
            train_precision += macro_precision
            train_recall += macro_recall
            train_f1 += macro_f1_score

        train_loss_avg = total_loss / len(train_loader)
        train_accuracy_avg = train_accuracy / len(train_loader)
        train_f1_avg = train_f1 / len(train_loader)
        train_precision_avg = train_precision / len(train_loader)
        train_recall_avg = train_recall / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_precision = 0
        val_recall = 0
        val_f1 = 0
        with torch.no_grad():
            for data in val_loader:
                y = data.y.long().to('cuda:1')
                
                classif_logits = model(data)                
                v_loss = criterion_classif(classif_logits, y)                    
                val_loss += v_loss.item()
                val_accuracy += calculate_accuracy(classif_logits, y)
                macro_precision, macro_recall, macro_f1_score = calculate_precision_recall_f1(classif_logits, y)
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

        os.makedirs(f'{model_path}', exist_ok = True)

        # if epoch % 20 == 0:
        #     torch.save(model.state_dict(), f"{model_path}/model_epoch_{epoch}.pth")
        # Early Stopping Check
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            if save_model_:
                #torch.save(model.state_dict(), ckpt_path)
                save_model(ckpt_path, f'best_{filename}', model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
        
        if save_model_:
            save_model(ckpt_path, f'last_{filename}', model)
    return results

def train_and_evaluate_encoder_decoder_FCFC_Temporal(model, path, filename, train_loader, val_loader, lamb, epochs, optimizer, criterion_recon, criterion_classif, early_stopping_patience, model_path, save_model_):
    ckpt_path = f'{path}/ckpt/'
    os.makedirs(ckpt_path, exist_ok=True)

    results = {
        'train_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'train_mse': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }

    # Inicialización para Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):  # Training and Validation loop
        model.train()
        total_loss = 0
        train_accuracy = 0
        train_precision = 0
        train_recall = 0
        train_f1 = 0
        train_mse = 0
        total_loss_class = 0

        for data in train_loader:
            optimizer.zero_grad()
            y = data.y.long().to('cuda:1')
            decoded, _, classif_logits, _, i= model(data)
            loss_recon = criterion_recon(decoded, data.fc[i+1].to('cuda:1'))
            loss_classif = criterion_classif(classif_logits, y)
            loss = loss_recon + lamb * loss_classif

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_class += loss_classif.item()
            
            train_accuracy += calculate_accuracy(classif_logits, y)
            macro_precision, macro_recall, macro_f1_score = calculate_precision_recall_f1(classif_logits, y)
            train_precision += macro_precision
            train_recall += macro_recall
            train_f1 += macro_f1_score
            train_mse += loss_recon.item()

        train_loss_class = total_loss_class / len(train_loader)
        train_loss_avg = total_loss / len(train_loader)
        train_accuracy_avg = train_accuracy / len(train_loader)
        train_f1_avg = train_f1 / len(train_loader)
        train_precision_avg = train_precision / len(train_loader)
        train_recall_avg = train_recall / len(train_loader)
        train_mse_avg = train_mse / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_precision = 0
        val_recall = 0
        val_f1 = 0
        with torch.no_grad():
            for data in val_loader:
                y = data.y.long().to('cuda:1')
                _, classif_logits, _ = model(data)
                v_loss_classif = criterion_classif(classif_logits, y)
                val_loss += v_loss_classif.item()
                val_accuracy += calculate_accuracy(classif_logits, y)
                macro_precision, macro_recall, macro_f1_score = calculate_precision_recall_f1(classif_logits, y)
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
        results['train_mse'].append(train_mse_avg)
        results['val_loss'].append(val_loss_avg)
        results['val_accuracy'].append(val_accuracy_avg)
        results['val_precision'].append(val_precision_avg)
        results['val_recall'].append(val_recall_avg)
        results['val_f1'].append(val_f1_avg)

        # print(f'Epoch {epoch+1}, Train Loss Classif: {train_loss_class:.2f}, Train Acc: {train_accuracy_avg:.2f}, Train MSE: {train_mse_avg:.3f}, Val Acc: {val_accuracy_avg:.2f}, Val MSE: {val_mse_avg:.3f}')
        print(f'Epoch {epoch+1}, Train Loss: {train_loss_avg:.2f}, Train Acc: {train_accuracy_avg:.2f}, Train Precision: {train_precision_avg:.2f}, Train Recall: {train_recall_avg:.2f},Train F1: {train_f1_avg:.2f}, Train MSE: {train_mse_avg:.2f}, Val Loss: {val_loss_avg:.2f}, Val Acc: {val_accuracy_avg:.2f}, Val Precision: {val_precision_avg:.2f}, Val Recall: {val_recall_avg:.2f}, Val F1: {val_f1_avg:.2f}')

        os.makedirs(f'{model_path}', exist_ok = True)
        # Check for early stopping
        if val_loss_avg < best_val_loss:
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
            if save_model_:
                #torch.save(model.state_dict(), ckpt_path)
                save_model(ckpt_path, f'best_{filename}', model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break

        if save_model_:
            #torch.save(model.state_dict(), f"{model_path}/final_model.pth")
            save_model(ckpt_path, f'final_{filename}', model)
    return results

def train_and_evaluate(model, path, filename, train_loader, val_loader, lamb, epochs, optimizer, criterion_recon, criterion_classif, early_stopping_patience, model_path, save_model_ ):

    if model.model_name in ['EncoderClassifierFCRandomTemporal']:
        results = train_and_evaluate_classifier(model, path, filename, train_loader, val_loader, epochs, optimizer, criterion_classif, early_stopping_patience, model_path, save_model_)
        return results
    
    if model.model_name in ['EncoderDecoderFCFCTemporal', 'EncoderDecoderFCFCMultipleTemporal']:
        results = train_and_evaluate_encoder_decoder_FCFC_Temporal(model, path, filename, train_loader, val_loader, lamb, epochs, optimizer, criterion_recon, criterion_classif, early_stopping_patience, model_path, save_model_)
        return results
