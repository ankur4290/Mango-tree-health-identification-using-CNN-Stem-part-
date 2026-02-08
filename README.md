# Mango Tree Health Identification Using CNN (Stem-Based)

## Overview
This project identifies the health condition of mango trees using images of the **stem part**.
A CNN-based transfer learning approach is used to classify stem images as **Healthy** or **Unhealthy**.

## Objective
- To classify mango tree stem images into healthy and unhealthy categories
- To compare different CNN models on a small dataset
- To study the effectiveness of transfer learning in agricultural image analysis

## Project Structure
Mango_Stem_Project/
├── images/
│ ├── healthy/
│ └── unhealthy/
├── mango_stem_labels/
│ └── labels.csv
├── train.py
├── requirements.txt
└── .gitignore

markdown
Copy code

## Dataset
- Approximately 150 mango tree stem images
- Two classes: Healthy and Unhealthy
- Dataset split: 80% training, 20% testing
- Labels are manually annotated

## Models Used
- MobileNet
- ResNet50
- VGG19

## Methodology
- Image preprocessing and data augmentation
- Transfer learning with pretrained CNN models
- Freezing base model layers
- Training for 5 epochs
- Saving model only when validation accuracy improves
- Model comparison using test accuracy

## Results
| Model      | Test Accuracy |
|-----------|---------------|
| MobileNet | ~89% |
| ResNet50  | ~52% |
| VGG19     | ~62% |

MobileNet achieved the best performance due to its lightweight architecture.

## How to Run
1. Install dependencies:
pip install -r requirements.txt

markdown
Copy code

2. Run the training script:
python train.py

markdown
Copy code

## Model Saving
- Models are saved only if validation accuracy improves
- Saved model files:
  - best_model_mobilenet.h5
  - best_model_resnet50.h5
  - best_model_vgg19.h5
- Trained models are excluded from GitHub using `.gitignore`

## Limitations
- Small dataset size
- Manual labeling may introduce noise
- Deep models require larger datasets for better performance

## Future Scope
- Increase dataset size
- Fine-tune pretrained layers
- Extend to multi-class disease classification

## Tools and Technologies
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- VS Code

## Academic Note
This project is developed as part of a B.Tech final-year