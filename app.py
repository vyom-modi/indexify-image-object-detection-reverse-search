# Import necessary libraries
import streamlit as st
from PIL import Image, ImageDraw
import os
import json

# Adding project-specific modules to system path
import sys
sys.path.append('scripts')
sys.path.append('utils')

from detect_objects import detect_objects
from reverse_search import store_features_in_vector_store, reverse_image_search, extract_features
from yolo_detection_graph import create_yolo_extraction_graph

detection_graph_spec = """
    name: 'yolo_detector'
    extraction_policies:
      - extractor: 'tensorlake/yolo-extractor'
        name: 'image_object_detection'
        input_params:
          model_name: 'yolov8n_trained.pt'
          conf: 0.25
          iou: 0.7
    """

reverse_image_graph_spec = """
    name: 'yolo_extractor'
    extraction_policies:
      - extractor: 'tensorlake/yolo-extractor'
        name: 'reverse_image_extractor'
        input_params:
            model_name: 'yolov8n.pt'
            conf: 0.25
            iou: 0.7
    """

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, objects):
    draw = ImageDraw.Draw(image)
    for obj in objects:
        content = json.loads(obj['content'])
        bbox = content['bbox']
        class_name = content['class']
        confidence = content['score']
        # Draw bounding box
        draw.rectangle(bbox, outline="red", width=3)
        # Add label with class name and confidence
        draw.text((bbox[0], bbox[1]), f"{class_name} {confidence:.2f}", fill="red")
    return image

# Initialize session state for storing features
if 'features' not in st.session_state:
    st.session_state['features'] = None

# Streamlit app title
st.title("Object Detection and Reverse Image Search")

# Selectbox to choose between object detection and reverse image search
mode = st.selectbox('Choose an Option', ('Detect Objects', 'Perform Reverse Image Search'))

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Save the uploaded image temporarily
    image_path = f"temp_{uploaded_image.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    # Display the original uploaded image in the left column

    if mode == "Detect Objects":
        st.write("")
        
        try: 
            # Attempt to create YOLO extraction graph
            create_yolo_extraction_graph(detection_graph_spec)
        except: pass
        
        st.write("#### **Detecting objects...**")
        # Create columns for displaying images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
        # Perform object detection on the uploaded image
        objects = detect_objects(image_path)

        # Draw bounding boxes on the image with detected objects
        image = Image.open(uploaded_image)
        image_with_boxes = draw_bounding_boxes(image, objects)

        # Display the image with bounding boxes in the right column
        with col2:
            st.image(image_with_boxes, caption="Image with Bounding Boxes", use_column_width=True)

        # Store features in vector store
        features = [{'content': obj['content']} for obj in objects]
        store_features_in_vector_store(features)
        st.session_state['features'] = features

    elif mode == "Perform Reverse Image Search":
        try: 
            # Attempt to create YOLO extraction graph
            create_yolo_extraction_graph(reverse_image_graph_spec)
        except: pass 

        print(image_path)

        column1, column2 = st.columns(2)
        with column1:    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        image_features = detect_objects(image_path)
        
        features = [{'content': feature['content']} for feature in image_features]

        features2 = image_features
        print(features2)

        store_features_in_vector_store(features2)
        # st.session_state['features'] = features
        # print(st.session_state['features'])

        # Button to trigger reverse image search
        if st.button("Perform Reverse Image Search"):
            if st.session_state['features']:
                features = features
                query_vector = json.loads(features[0]['content'].decode('utf-8'))['bbox'] + [json.loads(features[0]['content'].decode('utf-8'))['score']]
                search_result = reverse_image_search(query_vector)

                # Display search results
                st.write("## **Reverse Image Search Results:**")
                result_container = st.container()
                for result in search_result:
                    with result_container:
                        st.write(f"Image ID: {result.id}, Score: {result.score}")
                        st.write(result)
                        st.write('-'*40)
            else:
                st.error("Please run object detection first to store features.")

    # Clean up the temporary image file
    if os.path.exists(image_path):
        os.remove(image_path)
