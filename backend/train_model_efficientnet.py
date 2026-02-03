import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 10
EPOCHS_PHASE_2 = 5

TRAIN_DIR = "/kaggle/input/brain-tumor-mri-dataset/Training"
MODEL_SAVE_PATH = "model/brain_tumor_efficientnet.keras"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# ===============================
# DATA GENERATORS
# ===============================
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ===============================
# BASE MODEL (CNN)
# ===============================
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # PHASE 1 freeze

# ===============================
# CUSTOM HEAD
# ===============================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ===============================
# COMPILE – PHASE 1
# ===============================
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🚀 Phase 1 Training (Frozen Base)")
history_1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE_1
)

# ===============================
# FINE TUNING – PHASE 2
# ===============================
print("\n🔓 Unfreezing top layers for fine-tuning")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE_2
)

# ===============================
# SAVE MODEL
# ===============================
os.makedirs("model", exist_ok=True)
model.save(MODEL_SAVE_PATH)

print("\n✅ EfficientNet model saved at:", MODEL_SAVE_PATH)
