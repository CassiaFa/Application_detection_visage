import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from image_detect import *
import cv2
from PIL import Image, ImageEnhance
import os
import av

@st.cache
def load_image(img_path):
    """Charger une image"""
    im = Image.open(img_path)
    return im

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        img = visage_webcam(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

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
        # st.subheader("Détection de visage avec webcam")
        # run = st.checkbox('Run')
        # FRAME_WINDOW = st.image([])
        # camera = cv2.VideoCapture(0)

        # while run:
        #     _, frame = camera.read()
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     visage_webcam(frame)
        #     FRAME_WINDOW.image(frame)
        # else:
        #     st.write('Stopped')

        webrtc_streamer(key="example", video_processor_factory=VideoProcessor)


if __name__ == "__main__":
    main()