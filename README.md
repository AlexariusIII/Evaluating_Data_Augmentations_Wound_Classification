# WoundAug Benchmark

A comprehensive benchmark for wound image augmentation techniques in medical image classification.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

This repository contains the implementation and evaluation of various data augmentation techniques specifically designed for wound image classification. The benchmark includes geometric, photometric, elastic, and cutout augmentations, as well as combinations of these techniques. Our comprehensive study evaluates the effectiveness of different augmentation strategies on wound image classification tasks.

## Features

- Multiple augmentation strategies:
  - Geometric transformations (rotation, scaling, translation)
  - Photometric transformations (brightness, contrast, color jittering)
  - Elastic deformations (simulating tissue deformation)
  - Cutout augmentations (random masking)
  - Combined augmentation techniques (2-4 augmentations together)
- Support for multiple model architectures:
  - ResNet18
  - ConvNext Tiny
  - EfficientNetV2-S
  - Timm EfficientNetV2-S
- Comprehensive evaluation metrics
- Integration with Weights & Biases for experiment tracking
- Cross-validation support
- Hyperparameter optimization scripts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/woundaug_benchmark.git
cd woundaug_benchmark
```

2. Create and activate the conda environment:
```bash
conda env create -f env.yml
conda activate woundaug_env
```

## Usage

### Training

To train a model with specific augmentation techniques:

```bash
python train.py --dataset <dataset_name> --model <model_name> --augmentation <augmentation_type>
```

Available augmentation types:
- Single augmentations: `none`, `geometric`, `photometric`, `elastic`, `cutout`, `randaug`, `trivaug`
- Double augmentations: `geo_photo`, `geo_elastic`, `geo_cutout`, `photo_elastic`, `photo_cutout`, `elastic_cutout`
- Triple augmentations: `geo_photo_elastic`, `geo_photo_cutout`, `geo_elastic_cutout`, `photo_elastic_cutout`
- Quadruple augmentation: `geo_photo_elastic_cutout`

### Visualization

To visualize the augmentation effects:

```bash
jupyter notebook visualize_augmentations.ipynb
```

## Project Structure

- `train.py`: Main training script
- `woundaug_transforms.py`: Implementation of augmentation techniques
- `dataset.py`: Dataset loading and preprocessing
- `visualize_augmentations.ipynb`: Visualization notebook
- `optuna_search_*.py`: Hyperparameter optimization scripts
- `config.yaml`: Configuration file for training parameters

## Methodology

Our benchmark evaluates various data augmentation techniques for wound image classification:

1. **Geometric Transformations**
   - Random rotation
   - Random scaling
   - Random translation
   - Random horizontal flip

2. **Photometric Transformations**
   - Brightness adjustment
   - Contrast adjustment
   - Color jittering
   - Random erasing

3. **Elastic Deformations**
   - Simulated tissue deformation
   - Random elastic transformations

4. **Cutout Augmentations**
   - Random masking of image regions
   - Adaptive cutout sizes

5. **Combined Augmentations**
   - Double combinations (e.g., geometric + photometric)
   - Triple combinations
   - Quadruple combinations

## Results

Our comprehensive evaluation shows that:

1. Combined augmentations generally outperform single augmentations
2. The most effective combination varies by dataset and model architecture
3. Triple and quadruple augmentations can provide additional benefits but may increase training time

For detailed results and analysis, please refer to our paper.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{brehmer2024woundaug,
  title={WoundAug: A Comprehensive Benchmark for Wound Image Augmentation},
  author={Brehmer, Alexander and others},
  journal={[Journal Name]},
  year={2024},
  publisher={[Publisher]}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- We thank the contributors of the datasets used in this study
- Special thanks to the PyTorch and Albumentations teams for their excellent libraries
- This work was supported by [Your Institution/Organization]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue in the GitHub repository.
