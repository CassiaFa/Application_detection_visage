import streamlit as st
from image_detect import *
from PIL import Image, ImageEnhance
import os

@st.cache
def load_image(img_path):
    """Charger une image"""
    im = Image.open(img_path)
    return im

def main():
    """Application de detection de visage"""

    st.title("Apllication de détection de visage")
    st.text("Créer avec Streamlit et OpenCV")

    activities = ["Détection image", "Détection webcam"]
    choice = st.sidebar.selectbox("Que voulez-vous faire?", activities)

    if choice == "Détection image":
        st.subheader("Détection de visage sur image")

        image_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])

        if image_file is not None:
            
            if st.button("Détecter visage"):
                original_image = load_image(image_file)
                st.text("Visage détecté")
                img, tableau = detection_visage(original_image)
                st.image(img, use_column_width=True)
                st.dataframe(tableau)
            else:
                original_image = load_image(image_file)
                st.text("Image chargée")
                st.image(original_image, use_column_width=True)



    elif choice == "Détection webcam":
        st.subheader("Détection de visage avec webcam")
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            visage_webcam(frame)
            FRAME_WINDOW.image(frame)
        else:
            st.write('Stopped')


if __name__ == "__main__":
    main()