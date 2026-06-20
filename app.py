import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite Model
interpreter = tf.lite.Interpreter(
    model_path="brain_tumor_model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class Names
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# Title
st.title("🧠 Brain Tumor Classification")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # Display Image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    # Preprocessing
    img = image.resize((224, 224))

    img = np.array(img)

    img = img.astype(np.float32)

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    # Set Input
    interpreter.set_tensor(
        input_details[0]["index"],
        img
    )

    # Run Inference
    interpreter.invoke()

    # Get Output
    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    # Debug
    st.subheader("Debug Output")

    st.write("Raw Output:")
    st.write(output)

    st.write("Argmax:")
    st.write(np.argmax(output))

    # Prediction
    pred = np.argmax(output)

    confidence = np.max(output) * 100

    st.success(
        f"Prediction: {CLASS_NAMES[pred]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # Probability
    st.subheader("Probabilities")

    for i, cls in enumerate(CLASS_NAMES):
        st.write(
            f"{cls}: {float(output[0][i]) * 100:.2f}%"
        )
