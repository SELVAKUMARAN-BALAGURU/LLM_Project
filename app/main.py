import sys
import os

# Add parent directory to python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph import build_graph

def run_pipeline(file_path):
    print(f"\nStarting Multi-Agent Data Pipeline for: {file_path}")
    
    # 1. Build workflow
    workflow = build_graph()
    
    # 2. Initial state
    initial_state = {
        "file_path": file_path
    }
    
    # 3. Stream through the graph
    print("\n--- Pipeline Execution Started ---")
    for event in workflow.stream(initial_state):
        # We can just print the keys affected by the current node
        for key, value in event.items():
            pass # Logs are handled inside nodes
            
    print("\n--- Pipeline Execution Completed Iterating ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.main <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    run_pipeline(dataset_path)