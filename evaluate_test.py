import pandas as pd
from ultralytics import YOLO
import os
import time # Added for potential use in speed_mem
import torch # Added for potential use in speed_mem

# --- Configuration ---
DATA_YAML_PATH = r'C:\Users\LewPyt\Downloads\FYP.v3i.yolov11\data.yaml'  # Path to your dataset config file

# Paths to the best model weights
model_weights_paths = {
    'yolov12n_finetune': 'runs/finetune/yolov12n_finetune/weights/best.pt',
    'yolov12s_finetune': 'runs/finetune/yolov12s_finetune/weights/best.pt',
    'yolov12s_scratch': 'runs/scratch/yolov12s_scratch/weights/best.pt'
}
# --- End Configuration ---

# --- Helper Functions ---

def eval_test(ckpt_path, data_yaml, device=''):
    """
    Evaluates a YOLO model checkpoint on the test set,
    measures inference speed, and peak VRAM usage.
    """
    print(f"  Loading model from: {ckpt_path}")
    model = YOLO(ckpt_path)
    model.to(device) # Ensure model is on the correct device early

    # Reset VRAM counter before evaluation
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)
        print(f"  Reset peak VRAM stats for device: {device}")

    print(f"  Evaluating using data: {data_yaml}")
    # Evaluate on the 'test' split defined in data_yaml
    # Setting plots=False to avoid generating plots during evaluation
    # verbose=False to reduce console output during val
    r = model.val(data=data_yaml, split="test", imgsz=640, plots=False, verbose=False, device=device)

    # --- Extract Metrics ---
    map5095 = getattr(r.box, 'map', 0)    # mAP50-95
    map50 = getattr(r.box, 'map50', 0)  # mAP50
    precision = getattr(r.box, 'mp', 0) # precision
    recall = getattr(r.box, 'mr', 0)    # recall
    fitness = getattr(r, 'fitness', 0)  # Ultralytics “fitness” score
    print(f"  Metrics - mAP50-95: {map5095:.4f}, mAP50: {map50:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Fitness: {fitness:.4f}")

    # --- Extract Speed ---
    # Speed is reported in ms per image: [preprocess, inference, loss, postprocess]
    latency = r.speed.get('inference', 0) if hasattr(r, 'speed') and isinstance(r.speed, dict) else 0
    print(f"  Speed - Inference Latency: {latency:.2f} ms")

    # --- Measure VRAM ---
    vram = 0
    if device.startswith('cuda') and torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated(device=device) / (1024**2) # Convert bytes to MB
        print(f"  Peak VRAM allocated during eval: {vram:.2f} MB")
    else:
        print("  VRAM measurement skipped (not on CUDA device or CUDA not available).")

    return (map5095, map50, precision, recall, fitness, latency, vram)


# --- Main Execution ---

# # Check if data config file exists (Removed as requested, relying on model.val)
# if not os.path.exists(DATA_YAML_PATH):
#     print(f"Error: Data configuration file not found at '{DATA_YAML_PATH}'")
#     exit()

all_results = []
print("Starting model evaluation...")

for model_tag, weights_path in model_weights_paths.items():
    print(f"\nProcessing model: {model_tag}")
    if not os.path.exists(weights_path):
        print(f"Warning: Weights file not found for {model_tag} at '{weights_path}'. Skipping.")
        # Append placeholder data or skip entirely
        all_results.append({
            "Model Tag": model_tag,
            "mAP@0.5:0.95": 'N/A',
            "mAP@0.5": 'N/A',
            "Precision": 'N/A',
            "Recall": 'N/A',
            "Fitness": 'N/A',
            "Latency (ms)": 'N/A',
            "VRAM (MB)": 'N/A'
        })
        continue

    try:
        # Determine device automatically (cpu or cuda:0 if available)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"  Using device: {device}")

        # Evaluate performance metrics, speed, and memory
        map5095, map50, prec, rec, fitness, lat, vram = eval_test(weights_path, DATA_YAML_PATH, device=device)

        # Store results
        all_results.append({
            "Model Tag": model_tag,
            "mAP@0.5:0.95": map5095,
            "mAP@0.5": map50,
            "Precision": prec,
            "Recall": rec,
            "Fitness": fitness,
            "Latency (ms)": lat,
            "VRAM (MB)": vram
        })
        print(f"Finished processing {model_tag}.")

    except Exception as e:
        print(f"Error processing {model_tag}: {str(e)}")
        # Append error data
        all_results.append({
            "Model Tag": model_tag,
            "mAP@0.5:0.95": 'Error',
            "mAP@0.5": 'Error',
            "Precision": 'Error',
            "Recall": 'Error',
            "Fitness": 'Error',
            "Latency (ms)": 'Error',
            "VRAM (MB)": 'Error'
        })

# Convert results to DataFrame for better display
results_df = pd.DataFrame(all_results)

# Set display options for pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 4) # Format floats

# Print the results DataFrame
print("\n--- Combined Model Evaluation Results ---")
print(results_df.to_string(index=False)) # Print without index

# Optionally, save to CSV or Excel
# results_df.to_csv('model_evaluation_results.csv', index=False)
# results_df.to_excel('model_evaluation_results.xlsx', index=False)

print("\nEvaluation finished.")
