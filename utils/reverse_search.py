# reverse_search.py
import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from indexify import IndexifyClient

def extract_features(image_path):
    client = IndexifyClient()

    # Upload the image file
    image_id = client.upload_file("yolo_extractor", image_path)

    # Wait for the extraction to complete
    client.wait_for_extraction(image_id)

    # Retrieve the detected objects
    extracted_features = client.get_extracted_content(
        content_id=image_id,
        graph_name="yolo_extractor",
        policy_name="reverse_image_search"
    )

    return extracted_features


def store_features_in_vector_store(features):
    client = QdrantClient("http://localhost:6333")
    points = []

    for feature in features:
        feature_content = feature['content']
        feature_data = json.loads(feature_content.decode('utf-8'))
        bbox = feature_data['bbox']
        score = feature_data['score']

        # Create a vector representation combining bbox and score
        vector = bbox + [score]  # Assuming bbox is a list of numbers
        point_id = str(uuid.uuid4())
        points.append(models.PointStruct(id=point_id, vector=vector))
    
    if not points:
        print("No points to upsert, skipping request.")
    else:
        client.upsert(
            collection_name="image_search",
            points=points
        )

def reverse_image_search(query_vector, limit=5):
    client = QdrantClient("http://localhost:6333")
    search_result = client.search(
        collection_name="image_search",
        query_vector=query_vector,
        limit=limit  # Number of similar images to retrieve
    )
    return search_result
