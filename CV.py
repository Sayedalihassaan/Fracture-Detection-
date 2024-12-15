import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pyttsx3  # For text-to-speech
import os
import pickle
from io import BytesIO
from skimage.feature import graycoprops
from skimage.feature import graycomatrix
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

# Change working directory
os.chdir("D:\\FAI.3\\FAI.2.3.1\\Computer Vision")


# Preprocessing function
def preprocessing(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_resize = cv2.resize(gray, (224, 224))
    return img_resize



# Segmentation using contours
def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0) 
    _, thr = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    return cv2.bitwise_and(image, mask)



# Prepare image for feature extraction
def prepare_image(image, method):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == 'LBP':
        gray = cv2.equalizeHist(gray)
    return gray



# Feature extraction using Chain Code
def feature_extraction_chain_code(image):
    gray = prepare_image(image, 'chain_code')
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    chain_code = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        for i in range(1, len(contour)):
            dx = contour[i][0] - contour[i-1][0]
            dy = contour[i][1] - contour[i-1][1]
            direction = (np.arctan2(dy, dx) + np.pi) // (np.pi / 4)
            chain_code.append(int(direction))
    chain_code = np.pad(chain_code, (0, 2048 - len(chain_code)), 'constant')[:2048]
    return chain_code





# Feature extraction using LBP
def feature_extraction_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_scaled = np.uint8(lbp * (255.0 / np.max(lbp)))  # Scale to 0-255
    return lbp_scaled




# Speech function
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()



# Project description
project_description = """
Welcome to Fracture Detection
This project, Fracture Detection using Computer Vision, leverages deep learning techniques to classify medical images into two categories:
fracture and not fracture. By analyzing visual patterns, the system aims to assist healthcare professionals in identifying fractures more accurately 
and efficiently. The primary objective is to enhance diagnostic speed and reduce human error in medical imaging.
"""

# Project team 
project_team = """
Project Team includes: Sayed Ali, Abdallah Labeb, Rahma Mohamed, Rehab Mohamed, Ahmed Elnazer, Aseel Elfeky, Ahmed Shrif, Ahmed Saeed, Mostafa Osama, and Rahma Ashraf.
"""

# Load the model
model_path = r'binary_classifier.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)



if 'started' not in st.session_state or not st.session_state.started:
    st.markdown("""<div class="header" style="font-size: 32px; text-align: center;"><h2>😀 Welcome to Fracture Detection ❤️</h2></div><div class="content" style="font-size: 20px;"><h3>Project Description</h3><p style="color: #4CAF50 ; font-size: 22px;">Welcome to Fracture Detection. This project, <em>Fracture Detection using Computer Vision</em>, leverages deep learning techniques to classify medical images into two categories: <strong>fracture</strong> and <strong>not fracture</strong>. By analyzing visual patterns, the system aims to assist healthcare professionals in identifying fractures more accurately and efficiently. The primary objective is to enhance diagnostic speed and reduce human error in medical imaging. This tool provides a reliable, automated approach to support clinical decision-making.</p></div><div class="team" style="font-size: 20px;"><h2>Project Team:</h2><ul style="color: #4CAF50 ; font-size: 22px;"><li style="font-size: 20px;">Sayed Ali</li><li style="font-size: 20px;">Abdallah Labeb</li><li style="font-size: 20px;">Rahma Mohamed</li><li style="font-size: 20px;">Rehab Mohamed</li><li style="font-size: 20px;">Ahmed Elnazer</li><li style="font-size: 20px;">Aseel Elfeky</li><li style="font-size: 20px;">Ahmed Shrif</li><li style="font-size: 20px;">Ahmed Saeed</li><li style="font-size: 20px;">Mostafa Osama</li><li style="font-size: 20px;">Rahma Ashraf</li></ul></div>""", unsafe_allow_html=True)

    if 'read_description' not in st.session_state:
        st.session_state.read_description = True
        speak_text(project_description)

    if 'read_project_team' not in st.session_state:
        st.session_state.read_project_team = True
        speak_text(project_team)

    if st.button("Let's Go 😀"):
        st.session_state.started = True
        st.experimental_rerun()

else:
    st.title("😀Welcome to Fracture Detection")
    uploaded_file = st.file_uploader("Upload an image (JPEG/PNG)...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.markdown("##### 😀❤️Select Your Task: Segmentation, Feature Extraction, Classification.")
        option = st.radio("Choose an option", ("Segmentation", "Feature Extraction", "Classification"))

        if option == "Segmentation":
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            segmented_image = segment_image(image)
            st.markdown("## Segmented Image")
            st.image(segmented_image, caption="Segmented Image", use_column_width=True)

            # Prepare for image download
            segmented_image_pil = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
            buffered = BytesIO()
            segmented_image_pil.save(buffered, format="PNG")
            buffered.seek(0)
            st.download_button(
                label="Download Segmented Image",
                data=buffered,
                file_name="segmented_image.png",
                mime="image/png"
            )

        elif option == "Feature Extraction":
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

            optins = st.radio("Choose an option Feature Extraction", ("Chain Code", "LBP"))

            if optins == "Chain Code":
                image = Image.open(uploaded_file)
                image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                chain_code = feature_extraction_chain_code(image)

                # Optionally visualize the contours that were used to generate the chain code
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 127, 255, 0)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour_image = np.zeros_like(image)

                for contour in contours:
                    cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), 2)  # Drawing contours in green

                st.markdown("## Contour Visualization")
                st.image(contour_image, caption="Contours Visualized", use_column_width=True)

                chain_code_down = Image.fromarray(contour_image)

                buffered = BytesIO()
                chain_code_down.save(buffered, format="PNG")
                buffered.seek(0)

                st.download_button(
                    label="Download Chain Code Image",
                    data=buffered,
                    file_name="chain_code.png",
                    mime="image/png"
                )



            elif optins == "LBP":
                image = Image.open(uploaded_file)
                image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                
                # Extract LBP features
                lbp = feature_extraction_lbp(image)

                st.markdown("## LBP Features")
                
                # Display LBP image
                st.image(lbp, caption="LBP Features", use_column_width=True)

                lbp_down = Image.fromarray(lbp)  # Convert the numpy array back to an image

                # Create a BytesIO buffer to save the image
                buffered = BytesIO()
                lbp_down.save(buffered, format="PNG")
                buffered.seek(0)

                # Add download button for the LBP image
                st.download_button(
                    label="Download LBP Features Image",
                    data=buffered,
                    file_name="lbp.png",
                    mime="image/png"
                )
            
            


        elif option == "Classification":
            image = Image.open(uploaded_file)
            processed_image = preprocessing(image)
            processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
            processed_image = processed_image.astype('float32') / 255.0  # Normalize pixel values

            # Prediction
            prediction = model.predict(processed_image)

            # Classification logic
            if prediction[0][0] > 0.5:
                predicted_category = "Not Fracture"
                confidence = prediction[0][0] * 100
            else:
                predicted_category = "Fracture"
                confidence = (1 - prediction[0][0]) * 100

            # Display the result
            st.markdown("### Prediction Result")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"""
                <h4 style="color:green;">Predicted Class: {predicted_category}</h4>
                <h4 style="color:green;">Confidence: {confidence:.2f}%</h4>
            """, unsafe_allow_html=True)

    # Connect with me links
    st.markdown("""
        **Connect with me:**
        - [LinkedIn](https://www.linkedin.com/in/sayed-ali-482668262/)
        - [Kaggle](https://www.kaggle.com/engsaiedali/code)
        - [GitHub](https://github.com/Sayedalihassaan)
    """)

