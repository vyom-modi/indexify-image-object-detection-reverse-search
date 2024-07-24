import os
from indexify import IndexifyClient

def detect_objects(image_path):
    client = IndexifyClient()

    # Upload the image file
    content_id = client.upload_file("yolo_detector", image_path)

    # Wait for the extraction to complete
    client.wait_for_extraction(content_id)

    # Retrieve the detected objects
    detections = client.get_extracted_content(
        content_id=content_id,
        graph_name="yolo_detector",
        policy_name="image_object_detection"
    )

    return detections

