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
  <img src="./images/leopard.png" alt="CIFAR-100 Sample Images" width="600"/>
</p>
<p align="center">
  <img src="./images/leopard_image.png" alt="CIFAR-100 Sample Images" width="600"/>
</p>
<p align="center">
  <img src="./images/motorcycle_img.png" alt="CIFAR-100 Sample Images" width="600"/>
</p>
<p align="center">
  <img src="./images/train_image.png" alt="CIFAR-100 Sample Images" width="600"/>
</p>

