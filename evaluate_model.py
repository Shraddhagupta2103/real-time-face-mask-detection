from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = load_model("model/mask_detector_model.h5")

# Prepare the validation data generator (same setup as training)
IMG_SIZE = 224
BATCH_SIZE = 32
dataset_dir = "dataset/data"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Evaluate model
loss, accuracy = model.evaluate(val_gen)
print(f"✅ Validation Accuracy: {accuracy * 100:.2f}%")
