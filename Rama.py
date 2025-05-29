import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
from googletrans import Translator
from PIL import Image

# Constants and Configurations
CLASS_LABELS_50 = {
    0: "a", 1: "a lot", 2: "abdomen", 3: "able", 4: "about", 5: "above", 6: "accent", 7: "accept", 8: "accident",
    9: "accomplish", 10: "accountant", 11: "across", 12: "act", 13: "action", 14: "active", 15: "activity",
    16: "actor", 17: "adapt", 18: "add", 19: "address", 20: "adjective", 21: "adjust", 22: "admire", 23: "admit",
    24: "adopt", 25: "adult", 26: "advanced", 27: "advantage", 28: "adverb", 29: "affect", 30: "afraid",
    31: "africa", 32: "after", 33: "afternoon", 34: "again", 35: "against", 36: "age", 37: "agenda", 38: "ago",
    39: "agree", 40: "agreement", 41: "ahead", 42: "aid", 43: "aim", 44: "airplane", 45: "alarm", 46: "alcohol",
    47: "algebra", 48: "all", 49: "all day", 50: "allergy"
}

ASL_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space', 'nothing'
]

LANGUAGES = {
    'English': 'en', 'Hindi': 'hi', 'Bengali': 'bn', 'Spanish': 'es',
    'French': 'fr', 'German': 'de', 'Italian': 'it', 'Japanese': 'ja',
    'Korean': 'ko', 'Chinese (Simplified)': 'zh-cn', 'Russian': 'ru',
    'Arabic': 'ar', 'Telugu': 'te'
}

# Sign Language Emoji Mapping
sign_language_map = {
    'A': '👍', 'B': '👋', 'C': '👌', 'D': '👊', 'E': '✋', 'F': '✌️',
    'G': '🤞', 'H': '👏', 'I': '🤟', 'J': '🤙', 'K': '🖖', 'L': '👉',
    'M': '🤜', 'N': '🤛', 'O': '👌', 'P': '☝️', 'Q': '👇', 'R': '💪',
    'S': '🤲', 'T': '🙏', 'U': '✍️', 'V': '✊', 'W': '🤷', 'X': '🤞',
    'Y': '👆', 'Z': '🤜'
}


# Model Loading Functions
@st.cache_resource
def load_sign_language_model():
    model_path = r"/Users/venkatareddyvelagala/PycharmProjects/New/sign_language_cnn_model_50_classes.h5"
    return tf.keras.models.load_model(model_path)


@st.cache_resource
def load_asl_model():
    model_path = r"/Users/venkatareddyvelagala/PycharmProjects/New/Improved_ASL_Classifier.h5"
    return tf.keras.models.load_model(model_path)


# Utility Functions
def set_background_image(image_url, text_color="#FFFFFF", font_size="18px"):
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("{image_url}");
                background-size: cover;
                background-position : center;
                background-attachment: fixed;
                color: {text_color};
                font-family: 'Arial', sans-serif;
                font-size: {font_size};
            }}
            h1, h2, h3 {{ color: {text_color}; }}
            .button {{
                background-color: #007BFF;
                color: black;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .button:hover {{ background-color: #0056b3; }}
            .file-upload {{
                border: 2px dashed #007BFF;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                margin: 20px 0;
            }}
            .translation-box {{
                background-color: rgba(255, 255, 255, 0.9);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border: 2px solid #007BFF;
            }}
            .main-title {{
                text-align: center;
                font-size: 50px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #f8f9fa;
                text-shadow: 2px 2px 4px #000;
            }}
            .capture-area {{
                text-align: center;
                margin-top: 30px;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                font-size: 14px;
                color: #eeeeee;
            }}
        </style>
        """, unsafe_allow_html=True)


def translate_text(text, target_lang):
    translator = Translator()
    try:
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        return f"Translation error: {str(e)}"


def predict_sign_language(video_path, model):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            frame = cv2.resize(frame, (255, 255))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = gray_frame.reshape(1, 255, 255, 1)
            gray_frame = gray_frame / 255.0

            prediction = model.predict(gray_frame)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predictions.append(predicted_class)
        frame_count += 1

    cap.release()
    final_prediction = max(set(predictions), key=predictions.count)
    return CLASS_LABELS_50.get(final_prediction, "Unknown")


def predict_asl_image(image):
    model = load_asl_model()
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    predictions = model.predict(image)
    return predictions


# Page Functions
def login():
    set_background_image(
        "https://marketplace.canva.com/EAFCO6pfthY/1/0/1600w/canva-blue-green-watercolor-linktree-background-F2CyNS5sQdM.jpg",
        text_color="#2B63A9")
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "venkat" and password == "123456":
            st.success("Logged in successfully!", icon="✅")
            return True
        else:
            st.error("Incorrect username or password. Please try again.")
            return False


def sign_language_recognition_page():
    st.title("Sign Language Recognition")

    # Tab selection
    tab1, tab2 = st.tabs(["Video Recognition", "Real-time ASL"])

    with tab1:
        st.write("Upload a video of sign language to get the recognized text and its translation.")
        uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            st.video(video_path)

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Predict"):
                    model = load_sign_language_model()
                    prediction = predict_sign_language(video_path, model)
                    st.session_state.prediction = prediction
                    st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                    st.write("Recognized Word:")
                    st.subheader(prediction)
                    st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                selected_language = st.selectbox("Select Language for Translation", list(LANGUAGES.keys()))
                if st.button("Translate") and hasattr(st.session_state, 'prediction'):
                    translated_text = translate_text(st.session_state.prediction, LANGUAGES[selected_language])
                    st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                    st.write(f"Translation ({selected_language}):")
                    st.subheader(translated_text)
                    st.markdown('</div>', unsafe_allow_html=True)

            os.remove(video_path)

    with tab2:
        if "captured_text" not in st.session_state:
            st.session_state["captured_text"] = ""

        uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            st.markdown("<div class='capture-area'>", unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image = np.array(image)
            predictions = predict_asl_image(image)
            predicted_class = ASL_LABELS[np.argmax(predictions)]
            st.success(f"Predicted Sign: {predicted_class}")
            st.markdown("</div>", unsafe_allow_html=True)

        start_button = st.button("Start Webcam")
        stop_button = st.button("Stop Webcam")
        clear_button = st.button("Clear Text")

        if start_button:
            st.session_state["run_webcam"] = True
        if stop_button:
            st.session_state["run_webcam"] = False
        if clear_button:
            st.session_state["captured_text"] = ""

        if st.session_state.get("run_webcam", False):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()

            while st.session_state["run_webcam"]:
                ret, frame = cap.read()
                if not ret:
                    st.error("Webcam not accessible")
                    break

                frame = cv2.flip(frame, 1)
                cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
                roi = frame[100:400, 100:400]
                predictions = predict_asl_image(roi)
                predicted_class = ASL_LABELS[np.argmax(predictions)]
                confidence = np.max(predictions)

                cv2.putText(frame, f"{predicted_class}: {confidence:.2f}", (120, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                st.session_state["capture_frame"] = predicted_class
                if stop_button:
                    break
            cap.release()

        if st.button("Capture Sign"):
            if st.session_state.get("capture_frame") == 'space':
                st.session_state["captured_text"] += " "
            elif st.session_state.get("capture_frame") == 'del' and len(st.session_state["captured_text"]) > 0:
                st.session_state["captured_text"] = st.session_state["captured_text"][:-1]
            elif st.session_state.get("capture_frame") not in ['space', 'nothing', 'del']:
                st.session_state["captured_text"] += st.session_state["capture_frame"]

        st.text_area("Recognized Signs", value=st.session_state["captured_text"],
                     height=100, key="text_area", disabled=False)


def emoji_translation_page():
    st.title("Sign Language Emoji Translation")
    input_text = st.text_input("Enter the text (A-Z):", "")

    if input_text:
        try:
            signs = [sign_language_map[char.upper()] for char in input_text if char.upper() in sign_language_map]
            st.write("Sign Language Translation:")
            st.write("".join(signs))

            target_language = st.selectbox("Select a language for translation:", ["es", "fr", "de", "hi", "zh-cn"])
            if st.button("Translate"):
                translated_text = translate_text(input_text, target_language)
                st.write(f"Translated Text: {translated_text}")

        except Exception as e:
            st.error("Error occurred! Please enter valid letters (A-Z).")


def welcome_page():
    set_background_image(
        "https ://marketplace.canva.com/EAFCO6pfthY/1/0/1600w/canva-blue-green-watercolor-linktree-background-F2CyNS5sQdM.jpg",
        text_color="#2B63A9", font_size="20px")
    st.title("Welcome to the Sign Language Recognition App")
    st.write("""
        This Sign Language Recognition App is dedicated to breaking communication barriers for individuals 
        with hearing and speech impairments. Through this innovative application, users can upload videos 
        of sign language, and our advanced machine learning algorithms translate these signs into 
        comprehensible text.
    """)


def about_us_page():
    set_background_image(
        "https://marketplace.canva.com/EAFCO6pfthY/1/0/1600w/canva-blue-green-watercolor-linktree-background-F2CyNS5sQdM.jpg",
        text_color="#2B63A9", font_size="20px")
    st.title("About Us")
    st.write("""
        This Sign Language Recognition App is a groundbreaking tool aimed at promoting inclusivity 
        and accessibility in communication. Our mission is to develop intuitive, high-quality solutions 
        that make everyday interactions easier for individuals with communication challenges.
    """)


def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.session_state.logged_in = login()
    else:
        selected_page = st.sidebar.radio("Navigation",
                                         ["Welcome", "Sign Language Recognition", "Emoji Translation", "About Us"])

        if selected_page == "Welcome":
            welcome_page()
        elif selected_page == "Sign Language Recognition":
            sign_language_recognition_page()
        elif selected_page == "Emoji Translation":
            emoji_translation_page()
        elif selected_page == "About Us":
            about_us_page()


if __name__ == "__main__":
    main()