# will be used for torchvision fasterrcnn model zoo name
from sahi.utils.torchvision import TorchVisionTestConstants, download_torchvision_model

# import required functions, classes
from sahi.model import TorchVisionDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image


# set torchvision fasterrcnn model and config zoo name
download_torchvision_model()
model_path = TorchVisionTestConstants.FASTERCNN_MODEL_PATH,
config_path = TorchVisionTestConstants.FASTERCNN_CONFIG_ZOO_NAME,

# download test images into demo_data folder
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')

detection_model = TorchVisionDetectionModel(
    model_path=TorchVisionTestConstants.FASTERCNN_MODEL_PATH,
    config_path=TorchVisionTestConstants.FASTERCNN_CONFIG_ZOO_NAME,
    confidence_threshold=0.5,
    image_size=640,
    device="cpu", 
    load_at_init=True,
)


result = get_prediction("demo_data/small-vehicles1.jpeg", detection_model)
result = get_prediction(read_image("demo_data/small-vehicles1.jpeg"), detection_model)
result.export_visuals(export_dir="demo_data/")


import cv2

img_file = "demo_data/prediction_visual.png"
cv2.imshow("prediction_visual", cv2.imread(img_file))
cv2.waitKey(0)
cv2.destroyAllWindows()

