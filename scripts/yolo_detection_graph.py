from indexify import IndexifyClient, ExtractionGraph

def create_yolo_extraction_graph(extraction_graph_spec: str):
    client = IndexifyClient()
    extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
    client.create_extraction_graph(extraction_graph)
    print("YOLO extraction graph created successfully")

if __name__ == "__main__":
    create_yolo_extraction_graph()
