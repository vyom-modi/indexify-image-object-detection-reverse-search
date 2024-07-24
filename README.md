# YOLO Image Object Detection with Indexify

This cookbook demonstrates how to create an image object detection pipeline using Indexify and the tensorlake/yolo extractor. By the end of this guide, you'll have a pipeline capable of ingesting image files and detecting objects within them using the YOLO (You Only Look Once) model.

# Introduction

The image object detection pipeline will use the `tensorlake/yolo-extractor` extractor to process images and identify objects within them, providing bounding boxes, class names, and confidence scores for each detected object.

# Prerequisites

Before starting, ensure you have:

- A virtual environment with Python 3.9 or later
  ```shell
  conda create -n "myenv" python=3.9
  conda activate myenv
  pip install -r requirements.txt
  ```
- `pip` (Python package manager)
- Basic familiarity with Python and command-line interfaces

# Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/vyom-modi/indexify-image-object-detection-reverse-search.git
   cd image-object-detection-reverse-search
   ```
   
2. **Run Qdrant**

   Ensure that you have Qdrant running on your local machine. You can start Qdrant using Docker:

   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
 
## Installation

To get started, you need to install the required dependencies. Make sure you have Python installed on your machine.
   
### Install Indexify

First, install Indexify using the official installation script & start the server:

```bash
curl https://getindexify.ai | sh
./indexify server -d
```

This starts a long-running server that exposes ingestion and retrieval APIs to applications.

### Install Required Extractor

Next, install the YOLO extractor in a new terminal and start it:

```bash
indexify-extractor download tensorlake/yolo-extractor
indexify-extractor join-server
```

# Usage

## Creating YOLO Extraction Graph

First, create the YOLO extraction graph using the `yolo_detection_graph.py` script. This sets up the extraction graph using Indexify and the YOLO model for object detection.

```bash
python scripts/yolo_detection_graph.py
```

This script initializes the Indexify client and creates an extraction graph named `yolo_image_detector`. The graph uses the YOLO model (`yolov8n.pt`) with specified confidence and IoU thresholds for object detection.

## Object Detection

Use the `detect_objects.py` script to detect objects in images.

```bash
python utils/detect_objects.py
```

This script uploads the image to the `yolo_image_detector` extraction graph and retrieves the detected objects.

## Storing Features in Vector Store

The `reverse_search.py` script contains functions to store extracted features in the Qdrant vector store.

Run the script to process a dataset and store the features in the vector store.

```bash
python utils/reverse_search.py
```

## Reverse Image Search

You can perform a reverse image search using the same `reverse_search.py` script. It extracts features from a query image, searches the vector store, and retrieves similar images.

```bash
python utils/reverse_search.py
```

# File Structure
- `utils/detect_objects.py`: Script to detect objects in images using the YOLO extraction graph.
- `utils/reverse_search.py`: Script to store extracted features in the Qdrant vector store and perform reverse image search.
- `scripts/yolo_detection_graph.py`: Script to create the YOLO extraction graph.
- `data`: Directory to store image datasets.

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

# Conclusion

This example demonstrates the power and flexibility of using Indexify for image object detection:

1. **Scalability**: Indexify server can be deployed on a cloud and process numerous images uploaded into it. If any step in the pipeline fails, it automatically retries on another machine.
2. **Flexibility**: You can easily swap out components or adjust parameters to suit your specific needs.
3. **Integration**: The detected objects can be easily integrated into downstream tasks such as image analysis, indexing, or further processing.

