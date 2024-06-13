# Road Detection using UNet

This project implements a road detection system using a UNet model, a type of convolutional neural network that is widely used for image segmentation tasks. The system is designed to detect drivable areas on road images by generating binary segmentation masks.

## Project Structure

- **dataset**: Contains the training, validation, and test images along with their corresponding segmentation masks and lane markings. [Dataset Link](https://github.com/balnarendrasapa/road-detection/raw/master/datasets/dataset.zip)
  - **train/**
    - **images/**: Contains the training images.
    - **segments/**: Contains the segmentation masks for training images.
    - **lane/**: Contains the lane markings for training images.
  - **validation/**: Contains validation images and their corresponding masks and lanes.
  - **test/**: Contains test images and their corresponding masks and lanes.
- **unet_model.pth**: The saved UNet model weights.
- **scripts/**: Contains all the Python scripts for training, inference, and data handling.
  - **train.py**: Script for training the UNet model.
  - **inference.py**: Script for performing inference on a single image.
  - **dataset.py**: Defines the custom dataset class for loading images and masks.
  - **utils.py**: Contains utility functions such as image visualization.

## Installation

To get started with this project, you need to install the required dependencies. Check the unet_env create folder.

Ensure you have the following libraries installed:
- PyTorch
- OpenCV
- Albumentations
- Matplotlib
- PIL

## Data Preparation

The dataset directory should be structured as follows: (We used only segments and images.)

```
dataset/
├── train/
│   ├── images/
│   ├── segments/
│   ├── lane/
├── validation/
│   ├── images/
│   ├── segments/
│   ├── lane/
├── test/
│   ├── images/
│   ├── segments/
│   ├── lane/
```

- **images/**: Contains the raw images.
- **segments/**: Contains the binary segmentation masks indicating drivable areas.
- **lane/**: Contains the lane marking masks.

## Contributing

If you would like to contribute to this project, please fork the repository and create a pull request with your changes.
