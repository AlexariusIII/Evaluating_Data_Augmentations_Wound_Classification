import torch
from torchvision import datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from woundaug_transforms import WoundAug_Transforms
import os
from dataset import Classification_Dataset
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

def get_test_metrics(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total 
    f1 = f1_score(all_labels, all_preds, average='macro')  
    return accuracy, f1

def get_fold_mean_std(ckpt_fold_folder_path, test_loader, device, num_classes):
    accs = []
    f1s = []
    
    for fold_folder in os.listdir(ckpt_fold_folder_path):
        ckpt_path = os.path.join(ckpt_fold_folder_path, fold_folder, "best_model.pth")
        
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            continue

        # Initialize model (use a pre-trained model as an example)
        model = init_model_from_checkpoint(ckpt_path=ckpt_path, num_classes=num_classes)
        model = model.to(device)
        accuracy, f1 = get_test_metrics(model, test_loader, device)
        
        accs.append(accuracy)
        f1s.append(f1)
    
    mean_acc = round(np.mean(accs)*100, 2)
    std_acc = round(np.std(accs)*100, 2)
    mean_f1 = round(np.mean(f1s)*100, 2)
    std_f1 = round(np.std(f1s)*100, 2)
    return mean_acc, std_acc, mean_f1, std_f1

def init_model_from_checkpoint(ckpt_path, num_classes):
    model = models.efficientnet_v2_s(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(ckpt_path))
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test dataset
    dataset = 'medetec_4'
    dataset = 'azh_wound_dataset'
    data_dir = f'/home/jovyan/public-wound-datasets/{dataset}'
    image_dir = os.path.join(data_dir, 'images')
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    aug_transforms = WoundAug_Transforms(resize=(224, 224))
    test_dataset = Classification_Dataset(dataframe=test_df, image_dir=image_dir, transform=aug_transforms.get_val_transform())
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    num_classes = 4
    print(f"Num Classes: {num_classes}")

    # Load checkpoint folder
    ckpt_folder = '/home/jovyan/wound-aug-volume/woundaug_benchmark/checkpoints/'
    dataset_ckpt_folder = os.path.join(ckpt_folder, 'azh_wound_dataset')
    transform_names = []
    acc_means = []
    acc_stds = []
    f1_means = []
    f1_stds = []

    # Iterate through transformations
    for transform_variant in os.listdir(dataset_ckpt_folder):
        ckpt_fold_folder_path = os.path.join(dataset_ckpt_folder, transform_variant)
        
        if not os.path.isdir(ckpt_fold_folder_path):
            print(f"Skipping non-directory: {ckpt_fold_folder_path}")
            continue

        mean_acc, std_acc, mean_f1, std_f1 = get_fold_mean_std(ckpt_fold_folder_path, test_loader, device, num_classes)
        acc_means.append(mean_acc)
        acc_stds.append(std_acc)
        f1_means.append(mean_f1)
        f1_stds.append(std_f1)
        transform_names.append(transform_variant)

    # Save results to DataFrame
    results_df = pd.DataFrame({
        'Transform': transform_names,
        'Accuracy_Mean': acc_means,
        'Accuracy_Std': acc_stds,
        'F1_Mean': f1_means,
        'F1_Std': f1_stds
    })
    results_df = results_df.sort_values(by='Accuracy_Mean', ascending=False)
    results_df.to_csv(f"test_results/{dataset}_results.csv", index=False)

    print("Results saved successfully.")

if __name__ == "__main__":
    main()
