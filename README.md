# Binary Image Classification Project

A comprehensive deep learning project for binary image classification using Convolutional Neural Networks (CNN) with advanced data preprocessing, augmentation, and regularization techniques.

## Project Overview

This project implements an end-to-end binary image classification pipeline featuring automated data organization, comprehensive exploratory data analysis, advanced data preprocessing, and a regularized CNN model achieving ~86% accuracy on both training and test datasets.

## Key Features

- **Automated Data Management** - Directory creation and population based on train/test splits
- **Comprehensive EDA** - Multi-faceted analysis including class distribution and image characteristics
- **Advanced Preprocessing** - Image resizing, RGB conversion, and organized storage
- **Data Augmentation** - Enhanced dataset diversity to improve model generalization
- **Regularized Model** - CNN with L2 regularization and dropout for overfitting prevention
- **Performance Monitoring** - Early stopping and comprehensive evaluation metrics

## Project Workflow

### 1. Data Organization & Directory Setup

#### Directory Structure Creation
The project automatically creates and populates organized directories for efficient data management:

```
project/
├── train/
│   ├── class_0/
│   └── class_1/
├── test/
│   ├── class_0/
│   └── class_1/
└── processed/
    ├── resized_train/
    └── resized_test/
```

#### Key Operations:
- **Directory Creation**: Automated folder structure generation
- **Directory Marking**: Clear labeling system for easy navigation
- **Data Population**: Intelligent distribution based on specified train/test ratios
- **Size-based Organization**: Separate folders for different processing stages

### 2. Exploratory Data Analysis (EDA)

#### Image Count Analysis
- Statistical breakdown of total images per class
- Distribution verification across train/test splits
- Data balance assessment for model training optimization

#### Class Distribution
- Visual representation of class balance using bar plots
- Percentage breakdown of binary classification categories
- Identification of potential class imbalance issues

#### Sample Image Display
- Representative samples from each class
- Visual quality assessment
- Format and resolution verification

#### Image Size Distribution
- Analysis of original image dimensions
- Size variation statistics
- Memory usage calculations

#### Image Processing Pipeline
- **Resizing**: Standardization to 224x224 pixels for model compatibility
- **Subfolder Storage**: Organized storage of processed images
- **Format Verification**: Ensuring consistent image formats

#### Color Space Analysis
- **Grayscale vs RGB Detection**: Automatic identification of image color spaces
- **RGB Conversion**: Standardization to 3-channel RGB format
- **Color Distribution Analysis**: Statistical analysis of color characteristics

#### Data Augmentation
Implementation of various augmentation techniques:
- Rotation and flipping for geometric diversity
- Brightness and contrast adjustments
- Zoom and shift transformations
- Noise addition for robustness

### 3. Model Architecture & Training

#### Model Selection
Custom CNN architecture optimized for binary classification:

```python
Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

#### Architecture Highlights:
- **Progressive Feature Extraction**: Increasing filter sizes (32→64→128→128)
- **Spatial Reduction**: MaxPooling layers for dimension reduction
- **Regularization**: L2 regularization (0.01) on dense layer
- **Dropout**: 50% dropout rate for overfitting prevention
- **Binary Output**: Sigmoid activation for binary classification

#### Training Configuration
- **Loss Function**: Binary Crossentropy (optimal for binary classification)
- **Optimizer**: Adam with learning rate 1e-4 for stable convergence
- **Metrics**: Accuracy tracking for performance monitoring

#### Overfitting Prevention
- **EarlyStopping**: Automatic training halt when validation performance plateaus
- **Regularization**: L2 penalty on dense layer weights
- **Dropout**: Random neuron deactivation during training
- **Learning Rate**: Conservative rate (1e-4) for stable learning

### 4. Model Evaluation & Performance

#### Performance Metrics
- **Test Accuracy**: 86.31% - Strong generalization performance
- **Test Loss**: 0.38 - Low loss indicating good model fit
- **Train Accuracy**: 86.91% - Minimal overfitting (0.6% gap)

#### Visualization Components
- **Accuracy Plots**: Training vs validation accuracy over epochs
- **Loss Plots**: Training vs validation loss progression
- **Performance Comparison**: Side-by-side metric visualization
- **Confusion Matrix**: Detailed classification performance breakdown

#### Key Performance Indicators
- **Generalization**: Close train/test accuracy indicates good generalization
- **Stability**: Low loss values suggest stable model convergence
- **Reliability**: Consistent performance across different data splits

## Technical Specifications

### Model Architecture Details
- **Input Shape**: 224×224×3 (RGB images)
- **Total Parameters**: Optimized for efficient training
- **Activation Functions**: ReLU for hidden layers, Sigmoid for output
- **Regularization**: L2 + Dropout combination

### Data Processing Pipeline
- **Image Standardization**: Consistent 224×224 resolution
- **Color Space**: RGB format with 3 channels
- **Normalization**: Pixel value scaling for optimal training
- **Augmentation**: Real-time data enhancement

### Training Strategy
- **Batch Processing**: Efficient memory utilization
- **Early Stopping**: Automatic overfitting prevention
- **Validation Monitoring**: Real-time performance tracking
- **Checkpoint Saving**: Best model preservation

## Installation & Usage

### Prerequisites
```python
tensorflow>=2.0
keras
numpy
matplotlib
PIL (Pillow)
sklearn
```

### Execution Steps
1. **Data Preparation**: Upload dataset to Colab environment
2. **Directory Setup**: Run directory creation and population scripts
3. **EDA Execution**: Perform comprehensive data analysis
4. **Preprocessing**: Execute image resizing and format standardization
5. **Model Training**: Run training pipeline with early stopping
6. **Evaluation**: Generate performance plots and metrics

### Code Structure
```python
# Data Organization
create_directories()
populate_directories(test_size=0.2)

# EDA Pipeline
analyze_image_counts()
check_class_distribution()
display_sample_images()
analyze_image_sizes()

# Preprocessing
resize_and_store_images()
convert_to_rgb()
apply_data_augmentation()

# Model Training
model = create_model()
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')
train_with_early_stopping()

# Evaluation
evaluate_model()
plot_training_history()
```

## Results Summary

### Model Performance
- **Classification Accuracy**: 86%+ on both training and test sets
- **Low Overfitting**: Minimal gap between train and test performance
- **Stable Convergence**: Consistent loss reduction during training
- **Good Generalization**: Reliable performance on unseen data

### Key Achievements
- Automated end-to-end pipeline from raw data to trained model
- Comprehensive EDA providing deep dataset insights
- Effective regularization preventing overfitting
- Professional-grade code organization and documentation

## Future Enhancements

### Model Improvements
- Transfer learning with pre-trained models (VGG16, ResNet50)
- Ensemble methods for improved accuracy
- Advanced architectures (EfficientNet, Vision Transformers)

### Data Enhancements
- Advanced augmentation techniques (Mixup, CutMix)
- Synthetic data generation
- Cross-validation for robust evaluation

### Technical Upgrades
- Hyperparameter optimization (Optuna, Keras Tuner)
- Model compression and optimization
- Real-time inference capabilities

This binary image classification project demonstrates professional-level machine learning practices with comprehensive data analysis, robust model architecture, and thorough evaluation procedures.
