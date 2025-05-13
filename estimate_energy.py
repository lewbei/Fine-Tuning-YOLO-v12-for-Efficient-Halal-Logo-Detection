import pandas as pd
from pathlib import Path

# Constants for energy and CO2 calculations
AVG_POWER_WATTS = 250  # Average GPU power consumption in watts
EF_CO2_KWH = 0.40     # Malaysia grid emission factor (kg CO2/kWh)

def get_training_hours(results_path):
    """Extract total training time in hours from results.csv"""
    try:
        # Read results CSV and get last training time
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()
        if 'time' not in df.columns:
            print(f"Warning: 'time' column not found in {results_path}")
            return 0.0
            
        total_seconds = df['time'].iloc[-1]  # Get last time value
        return round(total_seconds / 3600, 2)  # Convert seconds to hours
        
    except Exception as e:
        print(f"Error processing {results_path}: {str(e)}")
        return 0.0

# Define paths to results.csv files
results_paths = {
    "YOLOv12s_scratch": "runs/scratch/yolov12s_scratch/results.csv",
    "YOLOv12s_finetune": "runs/finetune/yolov12s_finetune/results.csv", 
    "YOLOv12n_finetune": "runs/finetune/yolov12n_finetune/results.csv"
}

# Process each model's results
all_results = []
for name, path in results_paths.items():
    gpu_hours = get_training_hours(path)
    energy_kwh = round((gpu_hours * AVG_POWER_WATTS) / 1000, 2)
    co2_kg = round(energy_kwh * EF_CO2_KWH, 2)
    
    all_results.append({
        "Run": name,
        "GPU_hours": gpu_hours,
        "Avg_W (W)": AVG_POWER_WATTS,
        "Energy_kWh": energy_kwh,
        "CO2_kg": co2_kg
    })
    print(f"Processed {name}: {gpu_hours:.1f}h, {energy_kwh:.1f}kWh, {co2_kg:.1f}kg CO2")

# Create DataFrame and save to Excel
df_results = pd.DataFrame(all_results)
output_excel_path = "yolo_energy_results.xlsx"
df_results.to_excel(output_excel_path, index=False)
print(f"\nResults saved to {output_excel_path}")
