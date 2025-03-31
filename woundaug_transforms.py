import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter, map_coordinates
import numpy as np
import random
import torchvision.transforms.functional as F
import torch

class WoundAug_Transforms:
    def __init__(self, resize=(224, 224), mean=None, std=None):
        self.resize = resize
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]

    def random_zoom(self, img, zoom_range=(0.9, 1.1)):
        scale_factor = random.uniform(zoom_range[0], zoom_range[1])
        w, h = img.size
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        img = F.resize(img, (new_h, new_w))
        img = F.center_crop(img, (h, w))
        return img

    def add_gaussian_noise(self, tensor, mean=0., std=0.01):
        noise = torch.randn(tensor.size()) * std + mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)

    def randaugment_transform(self, num_ops=2, magnitude=9):
        return transforms.Compose([
            transforms.Resize(self.resize),                    # Resize the image to specified size
            transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),  # Apply RandAugment
            transforms.ToTensor(),                             # Convert to tensor
            transforms.Normalize(mean=self.mean, std=self.std) # Normalize
        ])

    def trivialaugment_transform(self):
        return transforms.Compose([
            transforms.Resize(self.resize),                    # Resize the image to specified size
            transforms.TrivialAugmentWide(),                   # Apply TrivialAugmentWide
            transforms.ToTensor(),                             # Convert to tensor
            transforms.Normalize(mean=self.mean, std=self.std) # Normalize
        ])

    def geometric_transform(self):
        """
        Create a geometric transformation pipeline based on provided parameters
        from Optuna optimization.
        """
        # Optuna parameters
        degrees = 14.66463432900851
        translate = (0.12072701989779873, 0.07813293284368938)
        scale = (0.8776342217401646, 1.0145155743568877)
        shear = 5.926234059170326
        use_resized_crop = True
        crop_scale = (0.690832277987517, 1.0)
        use_perspective = True
        distortion_scale = 0.4993889842603723
        use_rotation = False  
        rotation_degrees = 0  
        
        transformations = []
        
        # Optionally apply RandomResizedCrop
        if use_resized_crop:
            transformations.append(transforms.RandomResizedCrop(self.resize[0], scale=crop_scale))
        else:
            transformations.append(transforms.Resize(self.resize))
        # Always include basic horizontal and vertical flips
        transformations.append(transforms.RandomHorizontalFlip(p=0.5))
        transformations.append(transforms.RandomVerticalFlip(p=0.5))
        
        # Optionally add independent RandomRotation (skipped as use_rotation is False)
        if use_rotation:
            transformations.append(transforms.RandomRotation(rotation_degrees))
        
        # Add RandomAffine with dynamic parameters
        transformations.append(
            transforms.RandomAffine(
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear
            )
        )
        
        # Optionally add RandomPerspective
        if use_perspective:
            transformations.append(transforms.RandomPerspective(distortion_scale=distortion_scale, p=0.5))
        
        # Final transformations
        transformations.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        return transforms.Compose(transformations)



    def photometric_transform(self):
        """
        Create a photometric transformation pipeline based on Optuna optimization parameters.
        """
        # Optuna parameters
        brightness = 1.2923741568799973
        contrast = 1.2097514439353283
        saturation = 1.1825520869163884
        hue = 0.06758233698743349
        use_grayscale = True
        use_blur = False  # Set to False based on Optuna results
        blur_kernel = 3   # Default value, not used as use_blur is False
        use_sharpness = False  # Set to False based on Optuna results
        sharpness_factor = 1.0 # Default value, not used as use_sharpness is False
        use_equalize = True
        use_posterize = True
        posterize_bits = 5

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
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        return transforms.Compose(transformations)

    def elastic_transform(self):
        """
        Create an elastic transformation based on Optuna optimization parameters.
        """
        # Optuna parameters
        alpha = 34.79491454389255
        sigma = 5.9822016147394725
        
        def elastic_transform_image(img):
            img = np.array(img)
            if len(img.shape) == 2:  # Grayscale image (H, W)
                img = np.expand_dims(img, axis=-1)  # Add a channel dimension
            
            # Generate random displacement fields for x and y directions
            dx = gaussian_filter(np.random.randn(*img.shape[:2]), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter(np.random.randn(*img.shape[:2]), sigma, mode="constant", cval=0) * alpha
            
            # Generate a grid for coordinates
            x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
            
            # Map coordinates with the displacement
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
            # Apply the elastic transformation to each channel independently
            distorted_channels = [
                map_coordinates(img[..., i], indices, order=1, mode='reflect').reshape(img.shape[:2])
                for i in range(img.shape[-1])
            ]
            distorted_image = np.stack(distorted_channels, axis=-1)
            
            # If the original image was grayscale, remove the extra channel
            if distorted_image.shape[-1] == 1:
                distorted_image = distorted_image[..., 0]
            
            # Return the distorted image as PIL Image
            return transforms.ToPILImage()(distorted_image)
        
        return transforms.Compose([
            transforms.Resize(self.resize),              # Resize the image to specified size
            transforms.Lambda(elastic_transform_image),  # Apply the elastic transformation
            transforms.ToTensor(),                       # Convert to tensor
            transforms.Normalize(mean=self.mean, std=self.std)  # Normalize
        ])

    def cutout_transform(self):
        """
        Create a cutout (random erasing) transformation based on Optuna optimization parameters.
        """
        # Optuna parameters
        p = 0.7880842799502936
        scale = (0.0736687796371544, 0.23589488621910476)
        ratio = (0.6655672140882662, 2.704533304025621)
        value = 0

        return transforms.Compose([
            transforms.Resize(self.resize),                # Resize the image to specified size
            transforms.RandomHorizontalFlip(p=0.5),       # Random horizontal flip
            transforms.ToTensor(),                        # Convert to tensor
            transforms.Normalize(mean=self.mean, std=self.std),  # Normalize
            transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value)  # Random erase (cutout)
        ])


    # def geometric_transform(self):
    #     return transforms.Compose([
    #         transforms.Resize(self.resize),             
    #         transforms.RandomHorizontalFlip(p=0.5),     
    #         transforms.RandomVerticalFlip(p=0.5),       
    #         transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  
    #         transforms.RandomRotation(degrees=30),      
    #         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
    #         transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  
    #         transforms.Lambda(self.random_zoom),        
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=self.mean, std=self.std)  
    #     ])

    # def photometric_transform(self):
    #     return transforms.Compose([
    #         transforms.Resize(self.resize),              
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    #         transforms.RandomGrayscale(p=0.2),           
    #         transforms.GaussianBlur(kernel_size=3),      
    #         transforms.ToTensor(),                       
    #         #transforms.Lambda(lambda img: self.add_gaussian_noise(img)),  
    #         transforms.Normalize(mean=self.mean, std=self.std)  
    #     ])

    
    # def elastic_transform(self, alpha=20, sigma=5):
    #     alpha = np.random.uniform(alpha, alpha + 10)
    #     sigma = np.random.uniform(sigma, sigma + 2)

    #     def elastic_transform_image(img):
    #         img = np.array(img)
    #         if len(img.shape) == 2:  # Grayscale image (H, W)
    #             img = np.expand_dims(img, axis=-1)  # Add a channel dimension
            
    #         # Generate random displacement fields for x and y directions
    #         dx = gaussian_filter(np.random.randn(*img.shape[:2]), sigma, mode="constant", cval=0) * alpha
    #         dy = gaussian_filter(np.random.randn(*img.shape[:2]), sigma, mode="constant", cval=0) * alpha
            
    #         # Generate a grid for coordinates
    #         x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
            
    #         # Map coordinates with the displacement
    #         indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
    #         # Apply the elastic transformation to each channel independently
    #         distorted_channels = [
    #             map_coordinates(img[..., i], indices, order=1, mode='reflect').reshape(img.shape[:2])
    #             for i in range(img.shape[-1])
    #         ]
    #         distorted_image = np.stack(distorted_channels, axis=-1)
            
    #         # If the original image was grayscale, remove the extra channel
    #         if distorted_image.shape[-1] == 1:
    #             distorted_image = distorted_image[..., 0]
            
    #         # Return the distorted image as PIL Image
    #         return transforms.ToPILImage()(distorted_image)
        
    #     return transforms.Compose([
    #         transforms.Resize(self.resize),              # Resize the image to specified size
    #         transforms.Lambda(elastic_transform_image),  # Apply the elastic transformation
    #         transforms.ToTensor(),                       # Convert to tensor
    #         transforms.Normalize(mean=self.mean, std=self.std)  # Normalize
    #     ])

    # def cutout_transform(self):
    #     """
    #     Apply cutout or random erasing to the image.
        
    #     Returns:
    #     transforms.Compose: A composition of cutout transformations.
    #     """
    #     return transforms.Compose([
    #         transforms.Resize(self.resize),              # Resize the image to specified size
    #         transforms.RandomHorizontalFlip(p=0.5),      # Random horizontal flip
    #         transforms.ToTensor(),                       # Convert to tensor
    #         transforms.Normalize(mean=self.mean, std=self.std),  # Normalize
    #         transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')  # Random erase (cutout)
    #     ])

    def simple_double_transform(self,t1,t2):
        t1_transforms = [t for t in t1.transforms if not isinstance(t, (transforms.Normalize, transforms.Resize, transforms.ToTensor))]
        t2_transforms = [t for t in t2.transforms if not isinstance(t, (transforms.Normalize, transforms.Resize, transforms.ToTensor))]
        return transforms.Compose([
            transforms.Resize((224,224))]+
            t1_transforms+
            t2_transforms+
            [
            transforms.ToTensor(),                       
            transforms.Normalize(mean=self.mean, std=self.std),  # Normalize            
        ])
    
    def double_transform_with_cutout(self,t1):
        t1_transforms = [t for t in t1.transforms if not isinstance(t, (transforms.Normalize, transforms.Resize, transforms.ToTensor))]
        return transforms.Compose([
            transforms.Resize((224,224))]+
            t1_transforms+
            [
            transforms.ToTensor(),                       
            transforms.Normalize(mean=self.mean, std=self.std),  # Normalize
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')  # Random erase (cutout)
            ])

    def simple_triple_transform(self,t1,t2,t3):
        t1_transforms = [t for t in t1.transforms if not isinstance(t, (transforms.Normalize, transforms.Resize, transforms.ToTensor))]
        t2_transforms = [t for t in t2.transforms if not isinstance(t, (transforms.Normalize, transforms.Resize, transforms.ToTensor))]
        t3_transforms = [t for t in t3.transforms if not isinstance(t, (transforms.Normalize, transforms.Resize, transforms.ToTensor))]
        return transforms.Compose([
            transforms.Resize((224,224))]+
            t1_transforms+
            t2_transforms+
            t3_transforms+
            [
            transforms.ToTensor(),                       
            transforms.Normalize(mean=self.mean, std=self.std),  # Normalize            
        ])

    def triple_transform_with_cutout(self,t1,t2):
        t1_transforms = [t for t in t1.transforms if not isinstance(t, (transforms.Normalize, transforms.Resize, transforms.ToTensor))]
        t2_transforms = [t for t in t2.transforms if not isinstance(t, (transforms.Normalize, transforms.Resize, transforms.ToTensor))]

        return transforms.Compose([
            transforms.Resize((224,224))]+
            t1_transforms+
            t2_transforms+
            [
            transforms.ToTensor(),                       
            transforms.Normalize(mean=self.mean, std=self.std),  # Normalize
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')  # Random erase (cutout)
            ])
    
    def quadro_transform_withcutout(self,t1,t2,t3):
        t1_transforms = [t for t in t1.transforms if not isinstance(t, (transforms.Normalize, transforms.Resize, transforms.ToTensor))]
        t2_transforms = [t for t in t2.transforms if not isinstance(t, (transforms.Normalize, transforms.Resize, transforms.ToTensor))]
        t3_transforms = [t for t in t3.transforms if not isinstance(t, (transforms.Normalize, transforms.Resize, transforms.ToTensor))]

        return transforms.Compose([
            transforms.Resize((224,224))]+
            t1_transforms+
            t2_transforms+
            t3_transforms+
            [
            transforms.ToTensor(),                       
            transforms.Normalize(mean=self.mean, std=self.std),  # Normalize
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')  # Random erase (cutout)
            ])

        
    def get_val_transform(self):
        val_transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        return val_transform
