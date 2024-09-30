from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

def split_dataset_stratified(dataset):
    # Primero dividimos en entrenamiento+validación y prueba
    train_val, test, stratify_train_val, stratify_test = train_test_split(
        dataset, dataset.y, test_size=0.1, random_state=42, stratify=dataset.y
    )
     
    # Calcular el nuevo tamaño de validación basado en el conjunto reducido
    val_size_adjusted = 0.1/ (1 - 0.1)
    
    # Luego dividimos entrenamiento+validación en entrenamiento y validación
    train, val, _, _ = train_test_split(
        train_val, stratify_train_val, test_size=val_size_adjusted, random_state=42, stratify=stratify_train_val
    )
    
    return train, val, test


def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    return train_loader, val_loader, test_loader