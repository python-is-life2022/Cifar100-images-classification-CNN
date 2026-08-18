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
* Accuracy:
  
<p align="center">
  <img src="charts/train_1_accuracy.png" alt="Experiment 1 Training Curves" width="800"/>
</p>
* Loss:

<p align="center">
  <img src="charts/train_1_loss.png" alt="Experiment 1 loss Curves" width="800"/>
</p>

- **Observations**: 
  - The model experiences severe underfitting / plateauing early on due to the combination of `GlobalAveragePooling2D` directly after only two shallow conv layers and aggressive `Dropout` (up to 0.5) on small dense layers.
  - Tracking these curves serves as a visual benchmark for subsequent architectural iterations.

### Experiment 2: Deeper 4-Layer Convolutional Network

To enhance feature representation capacity for 100 fine-grained categories, the convolutional backbone was extended with a second convolutional block and trained for 60 epochs.

#### Architectural Modifications
- **Added Layer Block**: Added two consecutive `Conv2D(64, (3, 3), padding='same')` layers followed by a second `MaxPooling2D(2, 2)`.
- **Extended Training**: Epochs increased from 40 to 60.

| Metric / Split | Train Set | Test Set |
| :--- | :--- | :--- |
| **Accuracy** | **13.78%** | **12.37%** |
| **Loss** | **491.99** | **5.60e+02** |

#### Learning Curves
* Accuracy:
  
<p align="center">
  <img src="charts/train_2_accuracy.png" alt="Experiment 2 Training Accuracy Curves" width="800"/>
</p>
* Loss:

<p align="center">
  <img src="charts/train_2_loss.png" alt="Experiment 2 Training Loss Curves" width="800"/>
</p>

#### Key Takeaways
- Adding hierarchical depth allowed the network to learn higher-level spatial features compared to Experiment 1.
- *Note*: Ensure final evaluation is executed on normalized tensors (`X_test_norm`) to accurately reflect true test accuracy.

### 🧪 Experiment 3: Increasing Dense Capacity and Training Duration

#### 📌 Objective & Overview
In this experiment, the feature extraction backbone from Experiment 2 is retained (4 convolutional layers with 128 filters each), while the classification head is expanded with higher dense capacity (`Dense(256)` followed by `Dense(128)`). The training duration is extended to 100 epochs to observe long-term convergence and provide the network sufficient capacity for the 100 fine-grained classes.

---

#### ⚙️ Architecture & Hyperparameters

| Component / Layer | Details |
| :--- | :--- |
| **Input Shape** | `(32, 32, 3)` (Normalized float32) |
| **Conv Block 1** | $2\times$ `Conv2D(128, kernel_size=(3, 3), padding='same', relu)` $\rightarrow$ `MaxPooling2D(2, 2)` |
| **Conv Block 2** | $2\times$ `Conv2D(128, kernel_size=(3, 3), padding='same', relu)` $\rightarrow$ `MaxPooling2D(2, 2)` |
| **Pooling** | `GlobalAveragePooling2D()` |
| **Classifier Head** | `Dense(256, relu)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(128, relu)` $\rightarrow$ `Dropout(0.4)` |
| **Output Layer** | `Dense(100)` (Logits) |
| **Loss Function** | `SparseCategoricalCrossentropy(from_logits=True)` |
| **Optimizer** | `Adam(learning_rate=0.001)` |
| **Batch Size & Epochs** | `32` batch size \| `100` epochs \| `validation_split=0.2` |

---

#### 🔑 Key Modifications
- **Expanded Dense Capacity:** Increased classification head capacity using `Dense(256)` and `Dense(128)` to better handle high-dimensional feature representations.
- **Extended Training Budget:** Raised epochs from 60 to 100 to evaluate convergence and sustained learning trends.
- **Tuned Dropout Rates:** Set Dropout to 0.3 and 0.4 to prevent overfitting while avoiding excessive penalty on representational capacity.

---

#### 📊 Visualizations & Learning Curves

<p align="center">
  <img src="experiment_3_history.png" alt="Experiment 3 Training History" width="750"/>
</p>

---

#### 📈 Results & Evaluation

| Split | Loss | Accuracy |
| :--- | :--- | :--- |
| **Train** | *TBD* | *TBD*% |
| **Validation** | *TBD* | *TBD*% |
| **Test** | *TBD* | *TBD*% |

> **Summary Observation:**  
> *(Add your final notes here once evaluation finishes, e.g., comparison against Experiment 2 accuracy/loss convergence).*

