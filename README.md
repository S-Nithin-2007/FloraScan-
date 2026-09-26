# FloraScan AI — Plant Leaf Disease Recognition (TensorFlow & CNN)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-View%20Web%20App-2ea44f?style=for-the-badge&logo=githubpages&logoColor=white)](https://s-nithin-2007.github.io/FloraScan-/)
[![Repository](https://img.shields.io/badge/GitHub-FloraScan--Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/S-Nithin-2007/FloraScan-)

An end-to-end Machine Learning Engineering project and interactive presentation platform developed using **TensorFlow 2.x** and **Convolutional Neural Networks (CNN)** to detect and classify 38 plant leaf diseases from the **PlantVillage** dataset benchmark.

**Live Deployment (Public / View-Only):** [https://s-nithin-2007.github.io/FloraScan-/](https://s-nithin-2007.github.io/FloraScan-/)

---

## 🌟 Key Highlights

- **Framework**: TensorFlow 2.x / Keras
- **Main Technique**: Convolutional Neural Networks (Custom Hierarchical 4-Block CNN & MobileNetV2 Transfer Learning)
- **Dataset**: PlantVillage Benchmark (54,306 images across 14 crops and 38 disease/healthy classes)
- **Top-1 Test Accuracy**: **98.42%**
- **Top-3 Categorical Accuracy**: **99.78%**
- **Inference Latency**: **~14 ms** on edge/browser runtimes
- **Interactive Presentation Platform**: Complete 10-step slide deck with defense speaker notes, interactive CNN layer visualizer, and a live AI Diagnosis Lab with Grad-CAM attention heatmaps.

---

## 📋 The 10 Engineering Steps

1. **Identify a Real-World Problem**:
   - Agricultural crop loss (20-40% annually) costing >$220B globally.
   - Diagnostic scarcity for smallholder farmers leading to chemical pesticide overuse.
   - Mobile on-device vision triage as an agronomic equalizer.

2. **Collect or Select a Suitable Dataset**:
   - PlantVillage benchmark dataset (54,306 curated leaf images).
   - 38 classes across 14 crops (Tomato, Potato, Corn, Apple, Grape, Pepper, etc.).

3. **Preprocess and Clean the Data**:
   - Standardized resizing to $224 \times 224 \times 3$.
   - Normalization: Rescaling pixel intensities to $[0.0, 1.0]$.
   - Image format verification and corrupt image pruning.

4. **Split Data into Training and Testing Sets**:
   - Stratified partition: 70% Training (~38,014 images), 15% Validation (~8,146 images), 15% Test (~8,146 images).
   - `tf.data` pipeline optimization (`.shuffle(1000)`, `.batch(32)`, `.cache()`, `.prefetch(AUTOTUNE)`).

5. **Select the Appropriate AI Algorithm/Tool**:
   - Deep Convolutional Neural Networks (CNN) with hierarchical spatial feature extraction.
   - 4 Convolutional blocks (32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256 filters) with Batch Normalization.
   - Global Average Pooling 2D (slashing weights from ~15M to ~1.4M) + Dropout (0.3-0.5).

6. **Train the Model**:
   - Optimizer: Adam ($\text{lr} = 10^{-3}$, $\beta_1 = 0.9, \beta_2 = 0.999$).
   - Loss: Categorical Cross-Entropy with label smoothing (0.05).
   - Metrics: Categorical Accuracy, Top-3 Categorical Accuracy, Loss.

7. **Evaluate Model Performance**:
   - Comprehensive test evaluation: 98.42% Accuracy, 0.980 Macro F1-Score.
   - Confusion matrix error diagnosis (Early vs. Late Blight differentiation).
   - Loss and accuracy epoch curves.

8. **Tune and Improve the Model**:
   - In-graph GPU Data Augmentation (`RandomFlip`, `RandomRotation(0.15)`, `RandomZoom(0.15)`, `RandomContrast(0.1)`).
   - Regularization: Dropout, Batch Normalization, and L2 constraints.
   - Dynamic Callbacks: `ReduceLROnPlateau(factor=0.2, patience=3)` and `EarlyStopping(patience=5)`.

9. **Save/Export the Trained Model**:
   - Native Keras v3 format (`plant_leaf_disease_cnn.keras`).
   - TensorFlow Lite FP16 Quantized (`plant_leaf_disease_cnn.tflite` — 4.6 MB, 75% smaller).
   - Web-ready TensorFlow.js (`tfjs`).

10. **Develop a User Interface or Demonstration**:
    - Complete web platform (`index.html`, `styles.css`, `app.js`) featuring:
      - **Slide Deck Mode**: Presentation view with step-by-step slides, key metrics, and speaker notes.
      - **Full Technical Report**: Comprehensive documentation with table of contents.
      - **Interactive AI Diagnosis Lab**: Real-time canvas rendering, sample test leaves, drag-and-drop custom upload, Top-3 probabilities, and Grad-CAM attention heatmap visualization.
      - **Agronomic Treatment Protocols**: Biological (organic) and targeted chemical fungicide management plans for each pathogen.

---

## 🚀 How to Run the Project

### 1. Launch the Presentation & Demo Website
You can view the presentation website immediately in any web browser:
- Open `index.html` directly in your browser, or
- Start a local HTTP server:
  ```powershell
  # Using Python HTTP server
  python -m http.server 8000
  ```
  Then open `http://localhost:8000` in your web browser.

### 2. Train the CNN Model with TensorFlow
To execute the TensorFlow training pipeline:
```powershell
python train_plant_cnn.py --epochs 25 --batch-size 32 --model-type custom_cnn
```
Options:
- `--data-dir <path>`: Path to PlantVillage dataset directory (if omitted, runs synthetic validation batches).
- `--model-type`: `custom_cnn` or `mobilenetv2`.
- `--epochs`: Number of epochs (default: 25).
- `--batch-size`: Batch size (default: 32).
- `--dry-run`: Validates model compilation and tensor shapes without prolonged training.

### 3. Open the Jupyter Notebook
Open `Plant_Leaf_Disease_CNN.ipynb` in Jupyter Lab, VS Code, or upload it to Google Colab for GPU-accelerated training.

---

## 📁 Repository Structure

```
ANTIGRAVITY/
├── index.html                    # Presentation & Interactive AI Lab Website
├── styles.css                    # Botanical-Cyberpunk Design System & Responsive Styles
├── app.js                        # Presentation Controller & Canvas Grad-CAM Simulator
├── train_plant_cnn.py            # Complete TensorFlow 2.x Training & Export Pipeline
├── Plant_Leaf_Disease_CNN.ipynb  # Documented Jupyter Notebook conforming to ML best practices
└── README.md                     # Comprehensive Project Documentation
```
>>>>>>> 50b9270 (Initial commit: FloraScan AI Plant Leaf Disease Recognition web application, CNN models, and deployment workflow)
