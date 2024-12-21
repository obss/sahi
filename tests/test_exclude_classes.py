from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.file import download_from_url
from sahi.utils.yolov8 import download_yolov8s_model

# 1. Download the YOLOv8 model weights
yolov8_model_path = "models/yolov8s.pt"
download_yolov8s_model(yolov8_model_path)

# 2. Download sample test images
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
    "demo_data/terrain2.png",
)

# 3. Load the YOLOv8 detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",  # Model type (YOLOv8 in this case)
    model_path=yolov8_model_path,  # Path to model weights
    confidence_threshold=0.5,  # Confidence threshold for predictions
    device="cpu",  # Use "cuda" for GPU inference
)

# 4. Define the classes to exclude
exclude_classes_by_name = ["car"]

# 5. Demonstrate `get_prediction` with class exclusion
print("===== Testing `get_prediction` =====")
result = get_prediction(
    image="demo_data/small-vehicles1.jpeg",
    detection_model=detection_model,
    shift_amount=[0, 0],  # No shift applied
    full_shape=None,  # Full image shape is not provided
    postprocess=None,  # Postprocess disabled
    verbose=1,  # Enable verbose output
    exclude_classes_by_name=exclude_classes_by_name,  # Exclude 'car'
)

print("\nFiltered Results from `get_prediction` (First 5 Predictions):")
for obj in result.object_prediction_list[:5]:
    print(f"Class ID: {obj.category.id}, Class Name: {obj.category.name}, Score: {obj.score}")

# 6. Demonstrate `get_sliced_prediction` with and without filtering
print("\n===== Testing `get_sliced_prediction` (Without Filtering) =====")
result = get_sliced_prediction(
    image="demo_data/small-vehicles1.jpeg",
    detection_model=detection_model,
    slice_height=256,  # Slice height
    slice_width=256,  # Slice width
    overlap_height_ratio=0.2,  # Overlap height ratio
    overlap_width_ratio=0.2,  # Overlap width ratio
    verbose=1,  # Enable verbose output
)
print("\nNon-Filtered Results from `get_sliced_prediction` (First 5 Predictions):")
for obj in result.object_prediction_list[:5]:
    print(f"Class ID: {obj.category.id}, Class Name: {obj.category.name}, Score: {obj.score}")

print("\n===== Testing `get_sliced_prediction` (With Filtering) =====")
result = get_sliced_prediction(
    image="demo_data/small-vehicles1.jpeg",
    detection_model=detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    verbose=1,
    exclude_classes_by_name=exclude_classes_by_name,  # Exclude 'car'
)
print("\nFiltered Results from `get_sliced_prediction` (First 5 Predictions):")
for obj in result.object_prediction_list[:5]:
    print(f"Class ID: {obj.category.id}, Class Name: {obj.category.name}, Score: {obj.score}")

# 7. Demonstrate `predict` with filtering for a single image
print("\n===== Testing `predict` =====")
predict(
    detection_model=detection_model,
    source="demo_data/small-vehicles1.jpeg",  # Single image source
    project="runs/test_predict",  # Output project directory
    name="exclude_test",  # Run name
    verbose=1,  # Enable verbose output
    exclude_classes_by_name=exclude_classes_by_name,  # Exclude 'car'
)
print("\nFiltered results from `predict` saved in 'runs/test_predict/exclude_test'")
