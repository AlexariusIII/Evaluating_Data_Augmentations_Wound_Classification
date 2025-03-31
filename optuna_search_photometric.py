import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torchvision.models as models
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from dataset import Classification_Dataset
import numpy as np
import random
import optuna
import json
import argparse 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs, checkpoint_dir):
    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        val_loss, val_acc = validate_model(val_loader, model, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
        
        scheduler.step()

def validate_model(val_loader, model, criterion):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    val_acc = accuracy_score(all_labels, all_preds)
    return val_loss / len(val_loader), val_acc

def create_photometric_transform(brightness, contrast, saturation, hue, 
                                 use_grayscale=False, use_blur=False, blur_kernel=3,
                                 use_sharpness=False, sharpness_factor=1.0,
                                 use_equalize=False, use_posterize=False, posterize_bits=4):
    """
    Create a photometric transformation pipeline based on provided parameters.
    """
    transformations = []
    
    # Color jitter
    transformations.append(transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    ))

    # Optional transformations
    if use_grayscale:
        transformations.append(transforms.RandomGrayscale(p=0.2))
    if use_blur:
        transformations.append(transforms.GaussianBlur(kernel_size=blur_kernel))
    if use_sharpness:
        transformations.append(transforms.RandomAdjustSharpness(sharpness_factor, p=0.5))
    if use_equalize:
        transformations.append(transforms.RandomEqualize(p=0.5))
    if use_posterize:
        transformations.append(transforms.RandomPosterize(bits=posterize_bits))

    # Finalize transformations
    transformations.extend([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transformations)

def objective(trial):
    set_seed(42)
    dataset = 'azh_wound_dataset'
    data_dir = f'/home/jovyan/public-wound-datasets/{dataset}'
    image_dir = os.path.join(data_dir, 'images')
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    # Perform train/validation split
    fold = 0  # Example fold
    train_fold = train_df[train_df['fold'] != fold]
    val_fold = train_df[train_df['fold'] == fold]

    # Define photometric transformation hyperparameters using Optuna
    brightness = trial.suggest_float("brightness", 0.5, 1.5)
    contrast = trial.suggest_float("contrast", 0.5, 1.5)
    saturation = trial.suggest_float("saturation", 0.5, 1.5)
    hue = trial.suggest_float("hue", 0.0, 0.3)
    
    use_grayscale = trial.suggest_categorical("use_grayscale", [True, False])
    use_blur = trial.suggest_categorical("use_blur", [True, False])
    blur_kernel = trial.suggest_int("blur_kernel", 3, 7, step=2) if use_blur else 3

    use_sharpness = trial.suggest_categorical("use_sharpness", [True, False])
    sharpness_factor = trial.suggest_float("sharpness_factor", 0.5, 2.0) if use_sharpness else 1.0

    use_equalize = trial.suggest_categorical("use_equalize", [True, False])
    use_posterize = trial.suggest_categorical("use_posterize", [True, False])
    posterize_bits = trial.suggest_int("posterize_bits", 4, 8) if use_posterize else 4

    # Create the transformation directly in the script
    train_transform = create_photometric_transform(
        brightness, contrast, saturation, hue, use_grayscale, use_blur, blur_kernel,
        use_sharpness, sharpness_factor, use_equalize, use_posterize, posterize_bits
    )
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and loaders
    train_dataset = Classification_Dataset(dataframe=train_fold, image_dir=image_dir, transform=train_transform)
    val_dataset = Classification_Dataset(dataframe=val_fold, image_dir=image_dir, transform=val_transform)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    num_classes = len(train_df['label'].unique())
    model = models.efficientnet_v2_s(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # Set up training components
    lr = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    checkpoint_dir = "../checkpoints/optuna/photometric"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs, checkpoint_dir)
    
    # Validate using the best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    _, val_acc = validate_model(val_loader, model, criterion)
    
    return val_acc

def run_optuna_optimization():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)  # Adjust n_trials as needed

    # Print the best hyperparameters found
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save the best hyperparameters to a JSON file
    best_params = {
        "value": trial.value,
        "params": trial.params
    }
    json_save_path = os.path.join("/home/jovyan/wound-aug-volume/woundaug_benchmark/optuna_results","best_hyperparameters_photometric.json")
    with open(json_save_path, "w") as f:
        json.dump(best_params, f, indent=4)
    
    print("Best hyperparameters saved to best_photometric_hyperparameters.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization")
    parser.add_argument('--n_trials', type=int, default=10, help='Number of Optuna trials')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running on device: {device}")   
    n_trials = args.n_trials
    num_epochs = args.num_epochs

    run_optuna_optimization()
