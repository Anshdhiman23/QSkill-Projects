import os

import joblib
import numpy as np

import streamlit as st



st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)


st.title(
    "🌸 Iris Flower Classifier"
)

st.write(
    """
    Enter the measurements of an Iris flower
    and the machine-learning model will predict
    its species.
    """
)



MODEL_PATH = (
    "models/best_iris_model.joblib"
)


if not os.path.exists(MODEL_PATH):

    st.error(
        """
        Trained model not found.

        Run:

        `python train.py`

        first.
        """
    )

    st.stop()


model_data = joblib.load(
    MODEL_PATH
)


model = model_data["model"]

model_name = model_data[
    "model_name"
]

target_names = model_data[
    "target_names"
]



st.info(
    f"Model being used: **{model_name}**"
)



st.subheader(
    "Flower Measurements"
)


col1, col2 = st.columns(2)


with col1:

    sepal_length = st.number_input(
        "Sepal length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )


    sepal_width = st.number_input(
        "Sepal width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )


with col2:

    petal_length = st.number_input(
        "Petal length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )


    petal_width = st.number_input(
        "Petal width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )


if st.button(
    "Predict Species",
    type="primary",
    use_container_width=True
):


    features = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])



    prediction = model.predict(
        features
    )[0]



    probabilities = model.predict_proba(
        features
    )[0]


    predicted_species = (
        target_names[prediction]
    )


    st.success(
        f"""
        Predicted species:

        **{predicted_species.title()}** 🌸
        """
    )



    st.subheader(
        "Prediction Probabilities"
    )


    for species, probability in zip(
        target_names,
        probabilities
    ):

        st.write(
            f"**{species.title()}** — "
            f"{probability:.2%}"
        )

        st.progress(
            float(probability)
        )