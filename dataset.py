import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class Classification_Dataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'id' and 'label' columns.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.dataframe.iloc[idx]['id']
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_class_name(self, idx):
        return self.dataframe.iloc[idx]['class_name']