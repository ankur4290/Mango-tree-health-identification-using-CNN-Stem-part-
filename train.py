import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet, ResNet50, VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Optional: Clear session at start to free up RAM
tf.keras.backend.clear_session()

# 1. SETUP PATHS
# UPDATED: Using existing folder structure since automated move failed
IMAGES_PATH = 'images' # Folders: images/healthy and images/unhealthy
CSV_PATH = 'mango_stem_labels/mango_stem_classification.csv'

# 2. LOAD DATA
print("Loading CSV file...")
df = pd.read_csv(CSV_PATH)

# Categorize labels and fix paths
# The images are in subfolders: images/healthy and images/unhealthy
def fix_path(row):
    folder = 'healthy' if row['label'].lower() == 'healthy' else 'unhealthy'
    return os.path.join(folder, row['image_name'])

df['image_path'] = df.apply(fix_path, axis=1)
df['label'] = df['label'].str.capitalize() # Normalize to 'Healthy' and 'Unhealthy'

# Split into Training and Testing (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"Total images: {len(df)}")
print(f"Training images: {len(train_df)}")
print(f"Testing images: {len(test_df)}")

# 3. PREPROCESSING & DATA AUGMENTATION
# Resizing images to 224x224 (standard for these models)
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Augmentation for small dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Image Generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=IMAGES_PATH,
    x_col="image_path",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=IMAGES_PATH,
    x_col="image_path",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# 4. FUNCTION TO BUILD MODEL
def build_model(base_model_name):
    print(f"\nBuilding {base_model_name} model...")
    
    if base_model_name == 'MobileNet':
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the pretrained layers
    base_model.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x) # Binary classification
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 5. TRAINING MODELS
models_to_train = ['MobileNet', 'ResNet50', 'VGG19']
results = {}

for name in models_to_train:
    model = build_model(name)
    best_val_acc = 0.0  # Tracks best validation accuracy for this model
    save_filename = f"best_model_{name.lower()}.h5"
    
    print(f"Starting training for {name} (5 epochs manually)...")
    
    # Simple loop for 5 epochs to check for improvement
    for epoch in range(1, 6):
        print(f"\n--- {name} - Epoch {epoch}/5 ---")
        history = model.fit(
            train_generator,
            epochs=1,
            validation_data=test_generator,
            verbose=1
        )
        
        # Get validation accuracy of this specific epoch
        current_val_acc = history.history['val_accuracy'][0]
        
        # Saving Logic: Only save if validation accuracy improved
        if current_val_acc > best_val_acc:
            print(f"✅ Val Accuracy improved from {best_val_acc:.4f} to {current_val_acc:.4f}. Saving {save_filename}...")
            model.save(save_filename)
            best_val_acc = current_val_acc
        else:
            print(f"❌ Val Accuracy ({current_val_acc:.4f}) did not improve. Not saving.")

    # Final evaluation for accuracy comparison
    print(f"\nEvaluating {name} final performance...")
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    results[name] = {'Test_Accuracy': test_acc}

# 6. COMPARISON & SUMMARY
print("\n" + "="*30)
print(" MODEL COMPARISON SUMMARY")
print("="*30)
comparison_df = pd.DataFrame(results).T
print(comparison_df)

# Plotting Accuracy Comparison
comparison_df.plot(kind='bar', figsize=(10, 6))
plt.title('CNN Models Performance Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('model_comparison.png')
print("\nComparison plot saved as 'model_comparison.png'")
