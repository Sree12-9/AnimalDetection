
import json
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Animal Classification Using CNN",
    page_icon="🐾",
    layout="centered"
)

IMG_SIZE = (224, 224)

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(
        "animal_classifier.keras"
    )

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    return model, class_names

st.title("🐾 Animal Classification Using CNN")
st.write(
    "Upload a Cheetah, Jaguar, Leopard, Lion, or Tiger image."
)

try:
    model, class_names = load_model_and_labels()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded animal image",
            use_container_width=True
        )

        # Use the same 224x224 input size as training
        resized_image = image.resize(IMG_SIZE)
        image_array = np.array(
            resized_image, dtype=np.float32
        )

        # Model expects pixel values in the 0-255 range
        input_array = np.expand_dims(image_array, axis=0)

        with st.spinner("Predicting animal..."):
            probabilities = model.predict(
                input_array, verbose=0
            )[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_animal = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])

        st.subheader(f"Predicted Animal: {predicted_animal}")
        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.write("### Confidence by animal")

        sorted_indices = np.argsort(probabilities)[::-1]

        for index in sorted_indices:
            animal = class_names[index]
            score = float(probabilities[index])

            st.write(f"**{animal}: {score * 100:.2f}%**")
            st.progress(score)

        if confidence < 0.60:
            st.warning(
                "The prediction is uncertain. Try a clear image "
                "showing the animal's body and distinctive features."
            )

        st.caption(
            "Confidence is the model's estimated class probability, "
            "not a guarantee that the prediction is correct. "
            "This model only recognizes its five trained classes."
        )

    except Exception as e:
        st.error(f"Could not process this image: {e}")
