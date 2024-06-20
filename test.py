# prompt: generate button to upload image from computer and display it
"""
import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image file")

# Display the uploaded image
if uploaded_file is not None:
    # Read the image data
    file_bytes = uploaded_file.read()

    # Convert the image data to a PIL image
    from PIL import Image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img)
    results.print()

    # prompt: generate code to display the names of objects that was detected

    # Extract the detected object names from the results
    detected_objects = [model.names[result.xyxy[0][0][5].item()] for result in results.pandas().xyxy[0]]
    # Display the detected object names
    for obj_name, confidence in detected_objects:
        st.write(f"- {obj_name} (confidence: {confidence:.2f})")
"""
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Button to trigger analysis
    if st.button("Analyse Image"):
        # Read the image data
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Load the YOLOv5 model using local weights
        model = torch.hub.load('./yolov5', 'custom', path='yolov5/yolov5s.pt', source='local')

        # Perform object detection
        results = model(image)

        # Extract the detected object names and confidences from the results
        detected_objects = [(model.names[int(cls)], conf) for cls, conf in zip(results.xyxy[0][:, -1], results.xyxy[0][:, 4])]

        # Display the detected object names
        if detected_objects:
            st.write("Detected objects:")
            for obj_name, confidence in detected_objects:
                st.write(f"- {obj_name} (confidence: {confidence:.2f})")
        else:
            st.write("No objects detected.")
