import streamlit as st

from src.predict import predict_message, load_artifacts

st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Spam Mail Detector")
st.write("Paste an email or message and the trained NLP model will classify it as spam or ham.")

try:
    _, _, model_name = load_artifacts()
    st.caption(f"Current model: {model_name}")
except FileNotFoundError:
    st.warning("The model has not been trained yet. Run `python -m src.train` first.")

message = st.text_area(
    "Message",
    height=220,
    placeholder="Example: Congratulations! You have won a free prize...",
)

if st.button("Classify", type="primary", use_container_width=True):
    if not message.strip():
        st.error("Please enter a message.")
    else:
        try:
            result = predict_message(message)

            if result["label"] == "spam":
                st.error("🚨 SPAM")
            else:
                st.success("✅ HAM / NOT SPAM")

            if result["spam_probability"] is not None:
                st.metric(
                    "Estimated spam probability",
                    f"{result['spam_probability']:.1%}",
                )

            with st.expander("Preprocessed message"):
                from src.preprocessing import clean_text
                st.code(clean_text(message))

        except FileNotFoundError:
            st.error("Train the model first with `python -m src.train`.")
