# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import copy
import logging
import os
import random
import time
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
from PIL import Image

from sahi.utils.file import Path

logger = logging.getLogger("__name__")

IMAGE_EXTENSIONS_LOSSY = [".jpg", ".jpeg"]
IMAGE_EXTENSIONS_LOSSLESS = [".png", ".tif", ".tiff", ".bmp"]
IMAGE_EXTENSIONS = IMAGE_EXTENSIONS_LOSSY + IMAGE_EXTENSIONS_LOSSLESS
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".flv", ".avi", ".ts", ".mpg", ".mov", "wmv"]


class Colors:
    def __init__(self):
        hex = (
            "FF3838",
            "2C99A8",
            "FF701F",
            "6473FF",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "FF9D97",
            "00C2FF",
            "344593",
            "FFB21D",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex_to_rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, ind, bgr: bool = False):
        """
        Convert an index to a color code.

        Args:
            ind (int): The index to convert.
            bgr (bool, optional): Whether to return the color code in BGR format. Defaults to False.

        Returns:
            tuple: The color code in RGB or BGR format, depending on the value of `bgr`.
        """
        color_codes = self.palette[int(ind) % self.n]
        return (color_codes[2], color_codes[1], color_codes[0]) if bgr else color_codes

    @staticmethod
    def hex_to_rgb(hex_code):
        """
        Converts a hexadecimal color code to RGB format.

        Args:
            hex_code (str): The hexadecimal color code to convert.

        Returns:
            tuple: A tuple representing the RGB values in the order (R, G, B).
        """
        rgb = []
        for i in (0, 2, 4):
            rgb.append(int(hex_code[1 + i : 1 + i + 2], 16))
        return tuple(rgb)


def crop_object_predictions(
    image: np.ndarray,
    object_prediction_list,
    output_dir: str = "",
    file_name: str = "prediction_visual",
    export_format: str = "png",
):
    """
    Crops bounding boxes over the source image and exports it to the output folder.

    Args:
        image (np.ndarray): The source image to crop bounding boxes from.
        object_prediction_list: A list of object predictions.
        output_dir (str): The directory where the resulting visualizations will be exported. Defaults to an empty string.
        file_name (str): The name of the exported file. The exported file will be saved as `output_dir + file_name + ".png"`. Defaults to "prediction_visual".
        export_format (str): The format of the exported file. Can be specified as 'jpg' or 'png'. Defaults to "png".
    """
    # create output folder if not present
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # add bbox and mask to image if present
    for ind, object_prediction in enumerate(object_prediction_list):
        # deepcopy object_prediction_list so that the original is not altered
        object_prediction = object_prediction.deepcopy()
        bbox = object_prediction.bbox.to_xyxy()
        category_id = object_prediction.category.id
        # crop detections
        # deepcopy crops so that the original is not altered
        cropped_img = copy.deepcopy(
            image[
                int(bbox[1]) : int(bbox[3]),
                int(bbox[0]) : int(bbox[2]),
                :,
            ]
        )
        save_path = os.path.join(
            output_dir,
            file_name + "_box" + str(ind) + "_class" + str(category_id) + "." + export_format,
        )
        cv2.imwrite(save_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))


def convert_image_to(read_path, extension: str = "jpg", grayscale: bool = False):
    """
    Reads an image from the given path and saves it with the specified extension.

    Args:
        read_path (str): The path to the image file.
        extension (str, optional): The desired file extension for the saved image. Defaults to "jpg".
        grayscale (bool, optional): Whether to convert the image to grayscale. Defaults to False.
    """
    image = cv2.imread(read_path)
    pre, _ = os.path.splitext(read_path)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pre = pre + "_gray"
    save_path = pre + "." + extension
    cv2.imwrite(save_path, image)


def read_large_image(image_path: str):
    """
    Reads a large image from the specified image path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing the image data and a flag indicating whether cv2 was used to read the image.
            The image data is a numpy array representing the image in RGB format.
            The flag is True if cv2 was used, False otherwise.
    """
    use_cv2 = True
    # read image, cv2 fails on large files
    try:
        # convert to rgb (cv2 reads in bgr)
        img_cv2 = cv2.imread(image_path, 1)
        image0 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.debug(f"OpenCV failed reading image with error {e}, trying skimage instead")
        try:
            import skimage.io
        except ImportError:
            raise ImportError(
                'Please run "pip install -U scikit-image" to install scikit-image first for large image handling.'
            )
        image0 = skimage.io.imread(image_path, as_grey=False).astype(np.uint8)  # [::-1]
        use_cv2 = False
    return image0, use_cv2


def read_image(image_path: str) -> np.ndarray:
    """
    Loads image as a numpy array from the given path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image as a numpy array.
    """
    # read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return image
    return image


def read_image_as_pil(image: Union[Image.Image, str, np.ndarray], exif_fix: bool = False) -> Image.Image:
    """
    Loads an image as PIL.Image.Image.

    Args:
        image (Union[Image.Image, str, np.ndarray]): The image to be loaded. It can be an image path or URL (str),
            a numpy image (np.ndarray), or a PIL.Image object.
        exif_fix (bool, optional): Whether to apply an EXIF fix to the image. Defaults to False.

    Returns:
        PIL.Image.Image: The loaded image as a PIL.Image object.
    """
    # https://stackoverflow.com/questions/56174099/how-to-load-images-larger-than-max-image-pixels-with-pil
    Image.MAX_IMAGE_PIXELS = None

    if isinstance(image, Image.Image):
        image_pil = image
    elif isinstance(image, str):
        # read image if str image path is provided
        try:
            image_pil = Image.open(
                requests.get(image, stream=True).raw if str(image).startswith("http") else image  # type: ignore
            ).convert("RGB")
            if exif_fix:
                image_pil = exif_transpose(image_pil)
        except Exception as e:  # handle large/tiff image reading
            logger.debug(f"OpenCV failed reading image with error {e}, trying skimage instead")
            try:
                import skimage.io
            except ImportError:
                raise ImportError("Please run 'pip install -U scikit-image imagecodecs' for large image handling.")
            image_sk = skimage.io.imread(image).astype(np.uint8)
            if len(image_sk.shape) == 2:  # b&w
                image_pil = Image.fromarray(image_sk, mode="1")
            elif image_sk.shape[2] == 4:  # rgba
                image_pil = Image.fromarray(image_sk, mode="RGBA")
            elif image_sk.shape[2] == 3:  # rgb
                image_pil = Image.fromarray(image_sk, mode="RGB")
            else:
                raise TypeError(f"image with shape: {image_sk.shape[3]} is not supported.")
    elif isinstance(image, np.ndarray):
        if image.shape[0] < 5:  # image in CHW
            image = image[:, :, ::-1]
        image_pil = Image.fromarray(image)
    else:
        raise TypeError("read image with 'pillow' using 'Image.open()'")  # pyright: ignore[reportUnreachable]
    return image_pil


def select_random_color():
    """
    Selects a random color from a predefined list of colors.

    Returns:
        list: A list representing the RGB values of the selected color.

    """
    colors = [
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250],
        [50, 190, 190],
    ]
    return colors[random.randrange(0, 10)]


def apply_color_mask(image: np.ndarray, color: Tuple[int, int, int]):
    """
    Applies color mask to given input image.

    Args:
        image (np.ndarray): The input image to apply the color mask to.
        color (tuple): The RGB color tuple to use for the mask.

    Returns:
        np.ndarray: The resulting image with the applied color mask.
    """
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    (r[image == 1], g[image == 1], b[image == 1]) = color
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask


def get_video_reader(
    source: str,
    save_dir: str,
    frame_skip_interval: int,
    export_visual: bool = False,
    view_visual: bool = False,
) -> Tuple[Generator[Image.Image, None, None], Optional[cv2.VideoWriter], str, int]:
    """
    Creates OpenCV video capture object from given video file path.

    Args:
        source: Video file path
        save_dir: Video export directory
        frame_skip_interval: Frame skip interval
        export_visual: Set True if you want to export visuals
        view_visual: Set True if you want to render visual

    Returns:
        iterator: Pillow Image
        video_writer: cv2.VideoWriter
        video_file_name: video name with extension
    """
    # get video name with extension
    video_file_name = os.path.basename(source)
    # get video from video path
    video_capture = cv2.VideoCapture(source)

    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if view_visual:
        num_frames /= frame_skip_interval + 1
        num_frames = int(num_frames)

    def read_video_frame(video_capture, frame_skip_interval) -> Generator[Image.Image, None, None]:
        if view_visual:
            window_name = "Prediction of {}".format(str(video_file_name))
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            default_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imshow(window_name, default_image)

            while video_capture.isOpened:
                frame_num = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num + frame_skip_interval)

                k = cv2.waitKey(20)
                frame_num = video_capture.get(cv2.CAP_PROP_POS_FRAMES)

                if k == 27:
                    print(
                        "\n===========================Closing==========================="
                    )  # Exit the prediction, Key = Esc
                    exit()
                if k == 100:
                    frame_num += 100  # Skip 100 frames, Key = d
                if k == 97:
                    frame_num -= 100  # Prev 100 frames, Key = a
                if k == 103:
                    frame_num += 20  # Skip 20 frames, Key = g
                if k == 102:
                    frame_num -= 20  # Prev 20 frames, Key = f
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

                ret, frame = video_capture.read()
                if not ret:
                    print("\n=========================== Video Ended ===========================")
                    break
                yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        else:
            while video_capture.isOpened:
                frame_num = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num + frame_skip_interval)

                ret, frame = video_capture.read()
                if not ret:
                    print("\n=========================== Video Ended ===========================")
                    break
                yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if export_visual:
        # get video properties and create VideoWriter object
        if frame_skip_interval != 0:
            fps = video_capture.get(cv2.CAP_PROP_FPS)  # original fps of video
            # The fps of export video is increasing during view_image because frame is skipped
            fps = (
                fps / frame_skip_interval
            )  # How many time_interval equals to original fps. One time_interval skip x frames.
        else:
            fps = video_capture.get(cv2.CAP_PROP_FPS)

        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore[reportAttributeAccessIssue]
        video_writer = cv2.VideoWriter(os.path.join(save_dir, video_file_name), fourcc, fps, size)
    else:
        video_writer = None

    return read_video_frame(video_capture, frame_skip_interval), video_writer, video_file_name, num_frames


def visualize_prediction(
    image: np.ndarray,
    boxes: List[List],
    classes: List[str],
    masks: Optional[List[np.ndarray]] = None,
    rect_th: Optional[int] = None,
    text_size: Optional[float] = None,
    text_th: Optional[int] = None,
    color: Optional[tuple] = None,
    hide_labels: bool = False,
    output_dir: Optional[str] = None,
    file_name: Optional[str] = "prediction_visual",
):
    """
    Visualizes prediction classes, bounding boxes over the source image
    and exports it to output folder.

    Args:
        image (np.ndarray): The source image.
        boxes (List[List]): List of bounding boxes coordinates.
        classes (List[str]): List of class labels corresponding to each bounding box.
        masks (Optional[List[np.ndarray]], optional): List of masks corresponding to each bounding box. Defaults to None.
        rect_th (int, optional): Thickness of the bounding box rectangle. Defaults to None.
        text_size (float, optional): Size of the text for class labels. Defaults to None.
        text_th (int, optional): Thickness of the text for class labels. Defaults to None.
        color (tuple, optional): Color of the bounding box and text. Defaults to None.
        hide_labels (bool, optional): Whether to hide the class labels. Defaults to False.
        output_dir (Optional[str], optional): Output directory to save the visualization. Defaults to None.
        file_name (Optional[str], optional): File name for the saved visualization. Defaults to "prediction_visual".

    Returns:
        dict: A dictionary containing the visualized image and the elapsed time for the visualization process.
    """
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select predefined classwise color palette if not specified
    if color is None:
        colors = Colors()
    else:
        colors = None
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.003), 2)
    # set text_th for category names
    text_th = text_th or max(rect_th - 1, 1)
    # set text_size for category names
    text_size = text_size or rect_th / 3

    # add masks to image if present
    if masks is not None and color is None:
        logger.error("Cannot add mask, no color tuple given")
    elif masks is not None and color is not None:
        for mask in masks:
            # deepcopy mask so that original is not altered
            mask = copy.deepcopy(mask)
            # draw mask
            rgb_mask = apply_color_mask(np.squeeze(mask), color)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.6, 0)

    # add bboxes to image if present
    for box_indice in range(len(boxes)):
        # deepcopy boxso that original is not altered
        box = copy.deepcopy(boxes[box_indice])
        class_ = classes[box_indice]

        # set color
        if colors is not None:
            mycolor = colors(class_)
        elif color is not None:
            mycolor = color
        else:
            logger.error("color cannot be defined")
            continue

        # set bbox points
        point1, point2 = [int(box[0]), int(box[1])], [int(box[2]), int(box[3])]
        # visualize boxes
        cv2.rectangle(
            image,
            point1,
            point2,
            color=mycolor,
            thickness=rect_th,
        )

        if not hide_labels:
            # arange bounding box text location
            label = f"{class_}"
            box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[
                0
            ]  # label width, height
            outside = point1[1] - box_height - 3 >= 0  # label fits outside box
            point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
            # add bounding box text
            cv2.rectangle(image, point1, point2, color or (0, 0, 0), -1, cv2.LINE_AA)  # filled
            cv2.putText(
                image,
                label,
                (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
                0,
                text_size,
                (255, 255, 255),
                thickness=text_th,
            )
    if output_dir:
        # create output folder if not present
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # save inference result
        save_path = os.path.join(output_dir, (file_name or "unknown") + ".png")
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    elapsed_time = time.time() - elapsed_time
    return {"image": image, "elapsed_time": elapsed_time}


def visualize_object_predictions(
    image: np.ndarray,
    object_prediction_list,
    rect_th: Optional[int] = None,
    text_size: Optional[float] = None,
    text_th: Optional[int] = None,
    color: Optional[tuple] = None,
    hide_labels: bool = False,
    hide_conf: bool = False,
    output_dir: Optional[str] = None,
    file_name: Optional[str] = "prediction_visual",
    export_format: Optional[str] = "png",
):
    """
    Visualizes prediction category names, bounding boxes over the source image
    and exports it to output folder.

    Args:
        object_prediction_list: a list of prediction.ObjectPrediction
        rect_th: rectangle thickness
        text_size: size of the category name over box
        text_th: text thickness
        color: annotation color in the form: (0, 255, 0)
        hide_labels: hide labels
        hide_conf: hide confidence
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select predefined classwise color palette if not specified
    if color is None:
        colors = Colors()
    else:
        colors = None
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.003), 2)
    # set text_th for category names
    text_th = text_th or max(rect_th - 1, 1)
    # set text_size for category names
    text_size = text_size or rect_th / 3

    # add masks or obb polygons to image if present
    for object_prediction in object_prediction_list:
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()
        # arange label to be displayed
        label = f"{object_prediction.category.name}"
        if not hide_conf:
            label += f" {object_prediction.score.value:.2f}"
        # set color
        if colors is not None:
            color = colors(object_prediction.category.id)
        # visualize masks or obb polygons if present
        has_mask = object_prediction.mask is not None
        is_obb_pred = False
        if has_mask:
            segmentation = object_prediction.mask.segmentation
            if len(segmentation) == 1 and len(segmentation[0]) == 8:
                is_obb_pred = True

            if is_obb_pred:
                points = np.array(segmentation).reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(image, [points], isClosed=True, color=color or (0, 0, 0), thickness=rect_th)

                if not hide_labels:
                    lowest_point = points[points[:, :, 1].argmax()][0]
                    box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]
                    outside = lowest_point[1] - box_height - 3 >= 0
                    text_bg_point1 = (
                        lowest_point[0],
                        lowest_point[1] - box_height - 3 if outside else lowest_point[1] + 3,
                    )
                    text_bg_point2 = (lowest_point[0] + box_width, lowest_point[1])
                    cv2.rectangle(
                        image, text_bg_point1, text_bg_point2, color or (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA
                    )
                    cv2.putText(
                        image,
                        label,
                        (lowest_point[0], lowest_point[1] - 2 if outside else lowest_point[1] + box_height + 2),
                        0,
                        text_size,
                        (255, 255, 255),
                        thickness=text_th,
                    )
            else:
                # draw mask
                rgb_mask = apply_color_mask(object_prediction.mask.bool_mask, color or (0, 0, 0))
                image = cv2.addWeighted(image, 1, rgb_mask, 0.6, 0)

        # add bboxes to image if is_obb_pred=False
        if not is_obb_pred:
            bbox = object_prediction.bbox.to_xyxy()

            # set bbox points
            point1, point2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            # visualize boxes
            cv2.rectangle(
                image,
                point1,
                point2,
                color=color or (0, 0, 0),
                thickness=rect_th,
            )

            if not hide_labels:
                box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[
                    0
                ]  # label width, height
                outside = point1[1] - box_height - 3 >= 0  # label fits outside box
                point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
                # add bounding box text
                cv2.rectangle(image, point1, point2, color or (0, 0, 0), -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    image,
                    label,
                    (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
                    0,
                    text_size,
                    (255, 255, 255),
                    thickness=text_th,
                )

    # export if output_dir is present
    if output_dir is not None:
        # export image with predictions
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # save inference result
        save_path = str(Path(output_dir) / ((file_name or "") + "." + (export_format or "")))
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    elapsed_time = time.time() - elapsed_time
    return {"image": image, "elapsed_time": elapsed_time}


def get_coco_segmentation_from_bool_mask(bool_mask: np.ndarray) -> List[List[float]]:
    """
    Convert boolean mask to coco segmentation format
    [
        [x1, y1, x2, y2, x3, y3, ...],
        [x1, y1, x2, y2, x3, y3, ...],
        ...
    ]
    """
    # Generate polygons from mask
    mask = np.squeeze(bool_mask)
    mask = mask.astype(np.uint8)
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    polygons = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    # Convert polygon to coco segmentation
    coco_segmentation = []
    for polygon in polygons:
        segmentation = polygon.flatten().tolist()
        # at least 3 points needed for a polygon
        if len(segmentation) >= 6:
            coco_segmentation.append(segmentation)
    return coco_segmentation


def get_bool_mask_from_coco_segmentation(coco_segmentation: List[List[float]], width: int, height: int) -> np.ndarray:
    """
    Convert coco segmentation to 2D boolean mask of given height and width

    Parameters:
    - coco_segmentation: list of points representing the coco segmentation
    - width: width of the boolean mask
    - height: height of the boolean mask

    Returns:
    - bool_mask: 2D boolean mask of size (height, width)
    """
    size = [height, width]
    points = [np.array(point).reshape(-1, 2).round().astype(int) for point in coco_segmentation]
    bool_mask = np.zeros(size)
    bool_mask = cv2.fillPoly(bool_mask, points, (1.0,))
    bool_mask.astype(bool)
    return bool_mask


def get_bbox_from_bool_mask(bool_mask: np.ndarray) -> Optional[List[int]]:
    """
    Generate VOC bounding box [xmin, ymin, xmax, ymax] from given boolean mask.

    Args:
        bool_mask (np.ndarray): 2D boolean mask.

    Returns:
        Optional[List[int]]: VOC bounding box [xmin, ymin, xmax, ymax] or None if no bounding box is found.
    """
    rows = np.any(bool_mask, axis=1)
    cols = np.any(bool_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    width = xmax - xmin
    height = ymax - ymin

    if width == 0 or height == 0:
        return None

    return [xmin, ymin, xmax, ymax]


def get_bbox_from_coco_segmentation(coco_segmentation):
    """
    Generate voc box ([xmin, ymin, xmax, ymax]) from given coco segmentation
    """
    xs = []
    ys = []
    for segm in coco_segmentation:
        xs.extend(segm[::2])
        ys.extend(segm[1::2])
    if len(xs) == 0 or len(ys) == 0:
        return None
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    return [xmin, ymin, xmax, ymax]


def get_coco_segmentation_from_obb_points(obb_points: np.ndarray) -> List[List[float]]:
    """
    Convert OBB (Oriented Bounding Box) points to COCO polygon format.

    Args:
        obb_points: np.ndarray
            OBB points tensor from ultralytics.engine.results.OBB
            Shape: (4, 2) containing 4 points with (x,y) coordinates each

    Returns:
        List[List[float]]: Polygon points in COCO format
            [[x1, y1, x2, y2, x3, y3, x4, y4], [...], ...]
    """
    # Convert from (4,2) to [x1,y1,x2,y2,x3,y3,x4,y4] format
    points = obb_points.reshape(-1).tolist()

    # Create polygon from points and close it by repeating first point
    polygons = []
    # Add first point to end to close polygon
    closed_polygon = points + [points[0], points[1]]
    polygons.append(closed_polygon)

    return polygons


def normalize_numpy_image(image: np.ndarray):
    """
    Normalizes numpy image
    """
    return image / np.max(image)


def ipython_display(image: np.ndarray):
    """
    Displays numpy image in notebook.

    If input image is in range 0..1, please first multiply img by 255
    Assumes image is ndarray of shape [height, width, channels] where channels can be 1, 3 or 4
    """
    import IPython

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, ret = cv2.imencode(".png", image)
    i = IPython.display.Image(data=ret)  # type: ignore
    IPython.display.display(i)  # type: ignore


def exif_transpose(image: Image.Image) -> Image.Image:
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    Args:
        image (Image.Image): The image to transpose.

    Returns:
        Image.Image: The transposed image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.Transpose.FLIP_LEFT_RIGHT,
            3: Image.Transpose.ROTATE_180,
            4: Image.Transpose.FLIP_TOP_BOTTOM,
            5: Image.Transpose.TRANSPOSE,
            6: Image.Transpose.ROTATE_270,
            7: Image.Transpose.TRANSVERSE,
            8: Image.Transpose.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image
