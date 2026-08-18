# 🧠 CIFAR-100 Image Classification
### Building Robust CNN Architectures with TensorFlow
---
Dependencies & Imports
This project utilizes TensorFlow/Keras to build, train, and evaluate a Convolutional Neural Network (CNN) on the CIFAR-100 dataset. The following core modules are imported:

* Data Handling: numpy for efficient array operations.
* Model Architecture: Sequential for layer stacking and various layers (Conv2D, MaxPooling2D, Dense, etc.) to define the CNN structure.
* Optimization & Loss: Adam optimizer and SparseCategoricalCrossentropy for effective multi-class classification.
* Visualization: matplotlib.pyplot for monitoring training metrics and performance analysis.
  
## Data Preparation

The project uses the **CIFAR-100** dataset, which consists of 60,000 32x32 color images across 100 distinct classes.
```python
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
```
## Dataset Configuration
The project utilizes the CIFAR-100 dataset, containing 60,000 color images (32×32 resolution) categorized into 100 fine labels.

### Data Dimensions
The dataset is structured for CNN processing with the following tensor shapes:

- **Input Images**: `(N, 32, 32, 3)`  
  Representing `N` samples with height of 32, width of 32, and 3 color channels (RGB).
  - Training set size: 50,000
  - Testing set size: 10,000

- **Target Labels**: `(N, 1)`  
  Representing the single integer index for each class label.

*Note: The input pipeline expects these dimensions for optimal compatibility with the proposed CNN architecture.*
