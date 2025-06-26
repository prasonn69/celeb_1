import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image

# 1. Load image dataset
celebrity_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory='/Users/prason/Downloads/Sports-celebrity images',
    labels='inferred',
    label_mode='categorical',  # This gives one-hot encoded labels
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

# 2. Split into train and validation sets
train_size = int(0.8 * tf.data.experimental.cardinality(celebrity_data).numpy())
val_size = tf.data.experimental.cardinality(celebrity_data).numpy() - train_size

train_ds = celebrity_data.take(train_size)
val_ds = celebrity_data.skip(train_size)

# 3. Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# 4. Build CNN model
num_classes = len(celebrity_data.class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# 6. Save model
model.save('celebrity_model.h5')

# 7. Streamlit UI
st.title("Celebrity Prediction App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Batch of 1
    img_array = img_array / 255.0  # Rescale

    # Load model and predict
    model = tf.keras.models.load_model('celebrity_model.h5')
    predictions = model.predict(img_array)
    predicted_class = celebrity_data.class_names[np.argmax(predictions)]

    st.subheader(f"Prediction: {predicted_class}")
#accuracy
st.write(f"Model Accuracy: {history.history['accuracy'][-1]:.2f}")
#loss
st.write(f"Model Loss: {history.history['loss'][-1]:.2f}")