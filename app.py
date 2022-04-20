import streamlit as st
from PIL import Image
import torch


def main():
    new_title = '<p style="font-size: 42px;">SmartVisionAI!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    """
    )
    st.sidebar.title("Select Activity")
    choice  = st.sidebar.selectbox("MODE",("Object Detection(Image)","About"))

    file = None

    if choice == "Object Detection(Image)":

        read_me_0.empty()
        read_me.empty()
        st.title('SmartVisionAI detects prohibited items in real-time from X-ray Baggage Scanner')
        st.subheader("""
        SmartVisionAI is an Artificial Intelligence-based solution which ensures accuracy and efficiency in threat-detection at security checkpoints. With robust & fast algorithms at its core
        """)
        file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])

    elif choice == "About":
        print()

    if file!= None:    
        
        img1 = Image.open(file)
        st.image(img1, caption = "Uploaded Image")
        my_bar = st.progress(0)

        st.text("confidence threshold is the minimum score that the model will consider the prediction to be a true prediction (otherwise it will ignore this prediction entirely)")
        confThreshold =st.slider('Confidence', 0, 100, 50)

        st.text("IoU threshold is the minimum overlap between ground truth and prediction boxes for the prediction to be considered a true positive.")
        nmsThreshold= st.slider('IoU Threshold', 0, 100, 20)

        if st.button('Predict'):
            object_detection_image(img1, confThreshold, nmsThreshold, my_bar)

def object_detection_image(img1, confThreshold, nmsThreshold, my_bar):
    
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt',force_reload=True)
        model.conf = confThreshold/100
        model.iou = nmsThreshold/100

        results = model([img1])
        df = results.pandas().xyxy[0][['name',"confidence"]]
        df.columns=['Object Name','Confidence']

        st.write(df)

        st.subheader('Bar chart for confidence levels')
        st.bar_chart(df[["Confidence"]])

        results.save(save_dir="./")
        pred_img = results.imgs[0]
    
        st.image(pred_img, caption='Proccesed Image.')

        my_bar.progress(100)   

if __name__ == '__main__':
		main()	