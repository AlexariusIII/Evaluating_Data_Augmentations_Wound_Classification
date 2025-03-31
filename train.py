import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.models as models
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score
from dataset import Classification_Dataset
import wandb
from woundaug_transforms import WoundAug_Transforms
import numpy as np
import random
import argparse 
import timm
import yaml

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs, checkpoint_dir, config):
    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    last_model_path = os.path.join(checkpoint_dir, 'last_model.pth')

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

        # Calculate metrics using sklearn
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_loss, val_acc, val_f1 = validate_model(val_loader, model, criterion)

        # Log metrics to wandb if enabled
        if config.get('use_wandb', False):
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': running_loss / len(train_loader),
                'train_accuracy': train_acc,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            })
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}, '
              f'Training Accuracy: {train_acc:.4f}, Training F1: {train_f1:.4f}, '
              f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}')

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Saving the best model at epoch {epoch + 1} with validation accuracy {val_acc:.4f}")
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()
    torch.save(model.state_dict(), last_model_path)
    print(f"Saved the final model after epoch {num_epochs} at {last_model_path}")

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
    # Calculate metrics using sklearn
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    return val_loss / len(val_loader), val_acc, val_f1

def test_model(test_loader, model, criterion, config):
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    # Calculate metrics using sklearn
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    # Log the test performance to wandb if enabled
    if config.get('use_wandb', False):
        wandb.log({
            'test_loss': test_loss / len(test_loader),
            'test_accuracy': test_acc,
            'test_f1': test_f1
        })
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}')
    return test_loss / len(test_loader), test_acc, test_f1

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    set_seed(config.get('seed', 42))
    
    # Load data
    dataset = args.dataset
    data_dir = os.path.join(config['data_root'], dataset)
    image_dir = os.path.join(data_dir, 'images')
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'WoundAug'),
            entity=config.get('wandb_entity', None),
            config=config
        )
    
    #Init 5 fold cross validation
    for fold in set(train_df['fold'].to_list()):
        print(f"Training fold {fold}")
        train_fold = train_df[train_df['fold'] != fold]
        val_fold = train_df[train_df['fold'] == fold]   

        #Define single transforms transformations
        woundaug_transforms = WoundAug_Transforms()
        transform_dict = {
            'none': woundaug_transforms.get_val_transform(),
            'geometric': woundaug_transforms.geometric_transform(),
            'photometric': woundaug_transforms.photometric_transform(),
            'elastic': woundaug_transforms.elastic_transform(),
            'cutout': woundaug_transforms.cutout_transform(),
            'randaug' :  woundaug_transforms.randaugment_transform(),
            'trivaug': woundaug_transforms.trivialaugment_transform(),
        }
        transform_choice = args.augmentation

        if args.double_aug != None:
            transform_dict = {
                'geo_photo': woundaug_transforms.simple_double_transform(
                                    woundaug_transforms.geometric_transform(),
                                    woundaug_transforms.photometric_transform(),),
                'geo_elastic': woundaug_transforms.simple_double_transform(
                                    woundaug_transforms.geometric_transform(),
                                    woundaug_transforms.elastic_transform(),),
                'geo_cutout': woundaug_transforms.double_transform_with_cutout(
                                        woundaug_transforms.geometric_transform(),),
                'photo_elastic': woundaug_transforms.simple_double_transform(
                                    woundaug_transforms.photometric_transform(),
                                    woundaug_transforms.elastic_transform(),),
                'photo_cutout': woundaug_transforms.double_transform_with_cutout(
                                        woundaug_transforms.photometric_transform(),),
                'elastic_cutout': woundaug_transforms.double_transform_with_cutout(
                                        woundaug_transforms.elastic_transform(),),
            }
            transform_choice = args.double_aug
        
        if args.triple_aug != None:
            transform_dict = {
                'geo_photo_elastic': woundaug_transforms.simple_triple_transform(
                                    woundaug_transforms.geometric_transform(),
                                    woundaug_transforms.photometric_transform(),
                                    woundaug_transforms.elastic_transform(),
                                    ),
                'geo_photo_cutout': woundaug_transforms.triple_transform_with_cutout(
                                    woundaug_transforms.geometric_transform(),
                                    woundaug_transforms.photometric_transform(),),
                'geo_elastic_cutout': woundaug_transforms.triple_transform_with_cutout(
                                    woundaug_transforms.geometric_transform(),
                                    woundaug_transforms.elastic_transform(),),
                'photo_elastic_cutout': woundaug_transforms.triple_transform_with_cutout(
                                    woundaug_transforms.photometric_transform(),
                                    woundaug_transforms.elastic_transform(),),
            }
            transform_choice = args.triple_aug

        if args.quadro_aug != None:
            transform_dict = {
                'geo_photo_elastic_cutout': woundaug_transforms.quadro_transform_withcutout(
                                    woundaug_transforms.geometric_transform(),
                                    woundaug_transforms.photometric_transform(),
                                    woundaug_transforms.elastic_transform(),
                                    ),
            }
            transform_choice = args.quadro_aug

        train_transform = transform_dict[transform_choice]
        val_transform = woundaug_transforms.get_val_transform()
        
        train_dataset = Classification_Dataset(dataframe=train_fold, image_dir=image_dir, transform=train_transform)
        val_dataset = Classification_Dataset(dataframe=val_fold, image_dir=image_dir, transform=val_transform)
        test_dataset = Classification_Dataset(dataframe=test_df, image_dir=image_dir, transform=val_transform)

        batch_size = config.get('batch_size', 256)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize pre-trained model, loss function, and optimizer
        num_classes = len(train_df['label'].unique())
        model_dict = {
            'resnet18': models.resnet18(pretrained=True),
            'convnext_tiny': models.convnext_tiny(pretrained=True),
            'efficientV2_s': models.efficientnet_v2_s(pretrained=True),
            'timm_effnet_s' : timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)
        }
        model_choice = args.model
        model = model_dict[model_choice]

        # Modify the final layer according to the model architecture
        if model_choice == 'resnet18':
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_choice == 'convnext_tiny':
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        elif model_choice == 'efficientV2_s':
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_choice == 'timm_effnet_s':
            model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True, num_classes=num_classes)

        model = model.to(device)
        
        lr = config.get('learning_rate', 0.0001)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.get('weight_decay', 0.01))
        num_epochs = config.get('num_epochs', 50)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(config.get('checkpoint_dir', 'checkpoints'), 
                                    f"{dataset}_{model_choice}_{transform_choice}_fold{fold}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Train and validate the model
        train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, 
                   num_epochs, checkpoint_dir, config)
        
        # Test the model
        test_model(test_loader, model, criterion, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with wound image augmentations')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'convnext_tiny', 'efficientV2_s', 'timm_effnet_s'],
                      help='Model architecture to use')
    parser.add_argument('--augmentation', type=str, required=True,
                      choices=['none', 'geometric', 'photometric', 'elastic', 'cutout', 'randaug', 'trivaug'],
                      help='Single augmentation type')
    parser.add_argument('--double_aug', type=str, choices=['geo_photo', 'geo_elastic', 'geo_cutout', 
                      'photo_elastic', 'photo_cutout', 'elastic_cutout'],
                      help='Double augmentation type')
    parser.add_argument('--triple_aug', type=str, choices=['geo_photo_elastic', 'geo_photo_cutout',
                      'geo_elastic_cutout', 'photo_elastic_cutout'],
                      help='Triple augmentation type')
    parser.add_argument('--quadro_aug', type=str, choices=['geo_photo_elastic_cutout'],
                      help='Quadruple augmentation type')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main(args)