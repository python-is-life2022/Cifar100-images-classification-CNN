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

## Label Preprocessing
The CIFAR-100 target labels are originally loaded with shape `(N, 1)`.  
To simplify model training and evaluation, the labels are squeezed into a one-dimensional array with shape `(N,)` using `np.squeeze()`.

## Class Label Mapping and Image Visualization

CIFAR-100 provides numeric labels for its 100 image categories. The `fine_labels` list is used to map each label index to the corresponding class name, such as `apple`, `automobile`, `lion`, or `tiger`.

The following code displays a selected image from the training dataset and shows its class name as the plot title:
```python
i = int(input("Enter an image number: "))
plt.imshow(X_train[i])
plt.title(fine_labels[y_train[i]])
plt.savefig(f'{fine_labels[y_train[i]]}_image.png')
```
## Sample Dataset Images

Below is a sample of images from the CIFAR-100 training set with their corresponding fine-grained class labels:
<p align="center">
  <img src="./images/leopard_image.png" alt="CIFAR-100 Sample Images" width="600"/>
</p>
<p align="center">
  <img src="./images/motorcycle_img.png" alt="CIFAR-100 Sample Images" width="600"/>
</p>
<p align="center">
  <img src="./images/train_image.png" alt="CIFAR-100 Sample Images" width="600"/>
</p>
## Data Preprocessing & Normalization

Pixel intensities are originally represented as integers in the range `[0, 255]`. To optimize gradient descent and maintain numerical stability, the training and testing datasets are normalized to the `[0.0, 1.0]` range:
```python
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0
```
## Experiments History

### Experiment 1: Baseline CNN
This initial model aims to set a performance baseline using a custom CNN architecture.

#### Architecture
- **Layers**: 2x Conv2D filters, MaxPooling, GlobalAveragePooling, and a stack of Dense layers with Dropout (0.3 - 0.5) to prevent overfitting.
- **Training**: Trained for 40 epochs with a batch size of 32.

#### Results
| Metric | Result |
| :--- | :--- |
| **Train Accuracy** | 5.64% |
| **Test Accuracy** | 5.39% |
| **Train Loss** | 579.22 |

> **Technical Note**: The high loss observed during evaluation is due to an input scale mismatch (evaluating on raw `[0, 255]` pixel data instead of normalized `[0, 1]` data). This has been identified and will be addressed in subsequent experiments.
#### Learning Curves (Experiment 1)

<p align="center">
  <img src="charts/train_1_accuracy.png" alt="Experiment 1 Training Curves" width="800"/>
</p>

<p align="center">
  <img src="charts/train_1_loss.png" alt="Experiment 1 loss Curves" width="800"/>
</p>

- **Observations**: 
  - The model experiences severe underfitting / plateauing early on due to the combination of `GlobalAveragePooling2D` directly after only two shallow conv layers and aggressive `Dropout` (up to 0.5) on small dense layers.
  - Tracking these curves serves as a visual benchmark for subsequent architectural iterations.
### Experiment 2: Deeper CNN with 4 Convolutional Layers

In this experiment, network depth is expanded by introducing a second block of two `Conv2D(64)` layers and an additional `MaxPooling2D(2, 2)` layer. Training epochs are also increased from 40 to 60 to allow further feature extraction.

### Experiment 2: Deeper 4-Layer Convolutional Network

To enhance feature representation capacity for 100 fine-grained categories, the convolutional backbone was extended with a second convolutional block and trained for 60 epochs.

#### Architectural Modifications
- **Added Layer Block**: Added two consecutive `Conv2D(64, (3, 3), padding='same')` layers followed by a second `MaxPooling2D(2, 2)`.
- **Extended Training**: Epochs increased from 40 to 60.

#### Learning Curves

<p align="center">
  <img src="charts/train_2_accuracy.png" alt="Experiment 2 Training Accuracy Curves" width="800"/>
</p>
<p align="center">
  <img src="charts/train_2_Loss.png" alt="Experiment 2 Training Loss Curves" width="800"/>
</p>

#### Key Takeaways
- Adding hierarchical depth allowed the network to learn higher-level spatial features compared to Experiment 1.
- *Note*: Ensure final evaluation is executed on normalized tensors (`X_test_norm`) to accurately reflect true test accuracy.
