from __future__ import annotations

import copy
import os
import random
import time
from collections.abc import Generator
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

from sahi.logger import logger
from sahi.utils.file import Path

IMAGE_EXTENSIONS_LOSSY = [".jpg", ".jpeg"]
IMAGE_EXTENSIONS_LOSSLESS = [".png", ".tif", ".tiff", ".bmp"]
IMAGE_EXTENSIONS = IMAGE_EXTENSIONS_LOSSY + IMAGE_EXTENSIONS_LOSSLESS
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".flv", ".avi", ".ts", ".mpg", ".mov", "wmv"]


class Colors:
    def __init__(self):
        hex_colors = (
            "FF3838 2C99A8 FF701F 6473FF CFD231 48F90A 92CC17 3DDB86 1A9334 00D4BB "
            "FF9D97 00C2FF 344593 FFB21D 0018EC 8438FF 520085 CB38FF FF95C8 FF37C7"
        )

        self.palette = [self.hex_to_rgb(f"#{c}") for c in hex_colors.split()]
        self.n = len(self.palette)

    def __call__(self, ind, bgr: bool = False):
        """Convert an index to a color code.

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
        """Converts a hexadecimal color code to RGB format.

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
    """Crops bounding boxes over the source image and exports it to the output folder.

    Args:
        image (np.ndarray): The source image to crop bounding boxes from.
        object_prediction_list: A list of object predictions.
        output_dir (str): The directory where the resulting visualizations will be exported. Defaults to an empty string.
        file_name (str): The name of the exported file. The exported file will be saved as `output_dir + file_name + ".png"`. Defaults to "prediction_visual".
        export_format (str): The format of the exported file. Can be specified as 'jpg' or 'png'. Defaults to "png".
    """  # noqa

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
    """Reads an image from the given path and saves it with the specified extension.

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
    """Reads a large image from the specified image path.

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
        logger.error(f"OpenCV failed reading image with error {e}, trying skimage instead")
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
    """Loads image as a numpy array from the given path.

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


def read_image_as_pil(image: Image.Image | str | np.ndarray, exif_fix: bool = True) -> Image.Image:
    """Loads an image as PIL.Image.Image.

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
                BytesIO(requests.get(image, stream=True).content) if str(image).startswith("http") else image
            ).convert("RGB")
            if exif_fix:
                ImageOps.exif_transpose(image_pil, in_place=True)
        except Exception as e:  # handle large/tiff image reading
            logger.error(f"PIL failed reading image with error {e}, trying skimage instead")
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
        # check if image is in CHW format (Channels, Height, Width)
        # heuristic: 3 dimensions, first dim (channels) < 5, last dim (width) > 4
        if image.ndim == 3 and image.shape[0] < 5:  # image in CHW
            if image.shape[2] > 4:
                # convert CHW to HWC (Height, Width, Channels)
                image = np.transpose(image, (1, 2, 0))
        image_pil = Image.fromarray(image)
    else:
        raise TypeError("read image with 'pillow' using 'Image.open()'")
    return image_pil


def select_random_color():
    """Selects a random color from a predefined list of colors.

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


def apply_color_mask(image: np.ndarray, color: tuple[int, int, int]):
    """Applies color mask to given input image.

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
) -> tuple[Generator[Image.Image], cv2.VideoWriter | None, str, int]:
    """Creates OpenCV video capture object from given video file path.

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

    def read_video_frame(video_capture, frame_skip_interval) -> Generator[Image.Image]:
        if view_visual:
            window_name = f"Prediction of {video_file_name!s}"
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
    boxes: list[list],
    classes: list[str],
    masks: list[np.ndarray] | None = None,
    rect_th: int | None = None,
    text_size: float | None = None,
    text_th: int | None = None,
    color: tuple | None = None,
    hide_labels: bool = False,
    output_dir: str | None = None,
    file_name: str | None = "prediction_visual",
):
    """Visualizes prediction classes, bounding boxes over the source image and exports it to output folder.

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
    """  # noqa

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
    rect_th: int | None = None,
    text_size: float | None = None,
    text_th: int | None = None,
    color: tuple | None = None,
    hide_labels: bool = False,
    hide_conf: bool = False,
    output_dir: str | None = None,
    file_name: str | None = "prediction_visual",
    export_format: str | None = "png",
):
    """Visualizes prediction category names, bounding boxes over the source image and exports it to output folder.

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


def get_coco_segmentation_from_bool_mask(bool_mask: np.ndarray) -> list[list[float]]:
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


def get_bool_mask_from_coco_segmentation(coco_segmentation: list[list[float]], width: int, height: int) -> np.ndarray:
    """Convert coco segmentation to 2D boolean mask of given height and width.

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


def get_bbox_from_bool_mask(bool_mask: np.ndarray) -> list[int] | None:
    """Generate VOC bounding box [xmin, ymin, xmax, ymax] from given boolean mask.

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
    """Generate voc box ([xmin, ymin, xmax, ymax]) from given coco segmentation."""
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


def get_coco_segmentation_from_obb_points(obb_points: np.ndarray) -> list[list[float]]:
    """Convert OBB (Oriented Bounding Box) points to COCO polygon format.

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
    closed_polygon = [*points, points[0], points[1]]
    polygons.append(closed_polygon)

    return polygons


def normalize_numpy_image(image: np.ndarray):
    """Normalizes numpy image."""
    return image / np.max(image)


def ipython_display(image: np.ndarray):
    """Displays numpy image in notebook.

    If input image is in range 0..1, please first multiply img by 255
    Assumes image is ndarray of shape [height, width, channels] where channels can be 1, 3 or 4
    """
    import IPython

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, ret = cv2.imencode(".png", image)
    i = IPython.display.Image(data=ret)  # type: ignore
    IPython.display.display(i)  # type: ignore


# ============================================================================
# ENHANCED IMAGE PROCESSING UTILITIES
# ============================================================================


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Enhances local contrast and can improve detection in low-contrast images.
    
    Args:
        image: Input image (BGR or RGB)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Enhanced image with improved contrast
    """
    if len(image.shape) == 2:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    else:
        # Color image - apply to luminance channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    """Apply Gaussian blur for noise reduction.
    
    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Gaussian kernel standard deviation
    
    Returns:
        Blurred image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filter for edge-preserving smoothing.
    
    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
    
    Returns:
        Filtered image with edges preserved
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_unsharp_mask(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    """Apply unsharp masking for image sharpening.
    
    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel
        sigma: Gaussian kernel standard deviation
        amount: Strength of the sharpening effect
        threshold: Minimum brightness change to sharpen
    
    Returns:
        Sharpened image
    """
    blurred = apply_gaussian_blur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    
    return sharpened


def auto_gamma_correction(image: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
    """Automatically adjust gamma to achieve target mean brightness.
    
    Args:
        image: Input image
        target_mean: Target mean pixel value (0-255)
    
    Returns:
        Gamma-corrected image
    """
    # Convert to grayscale to calculate mean brightness
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    current_mean = np.mean(gray)
    
    # Calculate gamma value
    gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
    gamma = np.clip(gamma, 0.5, 2.5)  # Limit gamma range
    
    # Apply gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    
    return cv2.LUT(image, table)


def adaptive_threshold_image(image: np.ndarray, method: str = "gaussian", block_size: int = 11, c: int = 2) -> np.ndarray:
    """Apply adaptive thresholding for binarization.
    
    Args:
        image: Input grayscale image
        method: 'gaussian' or 'mean' for threshold calculation
        block_size: Size of pixel neighborhood (must be odd)
        c: Constant subtracted from weighted mean
    
    Returns:
        Binary image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if block_size % 2 == 0:
        block_size += 1
    
    if method == "gaussian":
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    else:
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)


def remove_background(image: np.ndarray, method: str = "mog2", learning_rate: float = 0.01) -> tuple:
    """Remove background using background subtraction.
    
    Args:
        image: Input image
        method: 'mog2' or 'knn' for background subtractor
        learning_rate: Learning rate for background model
    
    Returns:
        Tuple of (foreground_mask, foreground_image)
    """
    if method == "mog2":
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    else:
        bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    
    fg_mask = bg_subtractor.apply(image, learningRate=learning_rate)
    fg_image = cv2.bitwise_and(image, image, mask=fg_mask)
    
    return fg_mask, fg_image


def calculate_image_quality_score(image: np.ndarray) -> dict:
    """Calculate comprehensive image quality metrics.
    
    Args:
        image: Input image
    
    Returns:
        Dictionary containing quality metrics:
        - brightness: Mean brightness (0-255)
        - contrast: Standard deviation of brightness
        - sharpness: Variance of Laplacian (higher = sharper)
        - saturation: Mean saturation for color images
        - noise_estimate: Estimated noise level
    """
    # Convert to grayscale for some metrics
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Brightness
    brightness = np.mean(gray)
    
    # Contrast
    contrast = np.std(gray)
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    # Saturation (for color images)
    saturation = 0.0
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
    
    # Noise estimate (using median absolute deviation)
    h, w = gray.shape
    crop_size = min(h, w) // 4
    center_crop = gray[h//2 - crop_size:h//2 + crop_size, w//2 - crop_size:w//2 + crop_size]
    noise_estimate = np.median(np.absolute(center_crop - np.median(center_crop)))
    
    return {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "sharpness": float(sharpness),
        "saturation": float(saturation),
        "noise_estimate": float(noise_estimate),
    }


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Detect if image is blurry using Laplacian variance.
    
    Args:
        image: Input image
        threshold: Variance threshold (lower = more blurry)
    
    Returns:
        True if image is blurry, False otherwise
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def is_overexposed(image: np.ndarray, threshold: float = 0.05) -> bool:
    """Detect if image is overexposed.
    
    Args:
        image: Input image
        threshold: Ratio of pixels that are very bright (near 255)
    
    Returns:
        True if image is overexposed, False otherwise
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    bright_pixels = np.sum(gray > 250)
    total_pixels = gray.size
    bright_ratio = bright_pixels / total_pixels
    
    return bright_ratio > threshold


def is_underexposed(image: np.ndarray, threshold: float = 0.05) -> bool:
    """Detect if image is underexposed.
    
    Args:
        image: Input image
        threshold: Ratio of pixels that are very dark (near 0)
    
    Returns:
        True if image is underexposed, False otherwise
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    dark_pixels = np.sum(gray < 5)
    total_pixels = gray.size
    dark_ratio = dark_pixels / total_pixels
    
    return dark_ratio > threshold


def auto_enhance_image(image: np.ndarray, enhance_contrast: bool = True, enhance_sharpness: bool = True, denoise: bool = False) -> np.ndarray:
    """Automatically enhance image for better detection.
    
    Args:
        image: Input image
        enhance_contrast: Whether to apply CLAHE for contrast enhancement
        enhance_sharpness: Whether to apply unsharp masking
        denoise: Whether to apply denoising
    
    Returns:
        Enhanced image
    """
    enhanced = image.copy()
    
    # Denoise first if requested
    if denoise:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21) if len(enhanced.shape) == 3 else cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Enhance contrast
    if enhance_contrast:
        enhanced = apply_clahe(enhanced)
    
    # Enhance sharpness
    if enhance_sharpness:
        enhanced = apply_unsharp_mask(enhanced, amount=0.5)
    
    return enhanced


def apply_color_jitter(image: np.ndarray, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.1) -> np.ndarray:
    """Apply random color jitter for data augmentation.
    
    Args:
        image: Input image (BGR)
        brightness: Brightness jitter factor
        contrast: Contrast jitter factor
        saturation: Saturation jitter factor
        hue: Hue jitter factor
    
    Returns:
        Jittered image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Apply brightness jitter
    brightness_factor = 1.0 + random.uniform(-brightness, brightness)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    
    # Apply saturation jitter
    saturation_factor = 1.0 + random.uniform(-saturation, saturation)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    
    # Apply hue jitter
    hue_shift = random.uniform(-hue, hue) * 180
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 180)
    
    # Convert back to BGR
    jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Apply contrast jitter
    contrast_factor = 1.0 + random.uniform(-contrast, contrast)
    jittered = np.clip(jittered.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)
    
    return jittered


def apply_random_crop(image: np.ndarray, object_predictions: list, crop_ratio: float = 0.8) -> tuple:
    """Apply random crop while keeping all objects.
    
    Args:
        image: Input image
        object_predictions: List of ObjectPrediction instances
        crop_ratio: Ratio of crop size to original size
    
    Returns:
        Tuple of (cropped_image, adjusted_predictions)
    """
    h, w = image.shape[:2]
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    
    # Find bounding box of all objects
    if object_predictions:
        all_bboxes = np.array([pred.bbox.to_xyxy() for pred in object_predictions])
        min_x = np.min(all_bboxes[:, 0])
        min_y = np.min(all_bboxes[:, 1])
        max_x = np.max(all_bboxes[:, 2])
        max_y = np.max(all_bboxes[:, 3])
        
        # Ensure crop contains all objects
        max_start_x = max(0, min(int(min_x), w - crop_w))
        max_start_y = max(0, min(int(min_y), h - crop_h))
        min_end_x = max(crop_w, min(int(max_x), w))
        min_end_y = max(crop_h, min(int(max_y), h))
        
        start_x = random.randint(max(0, min_end_x - crop_w), min(max_start_x, w - crop_w))
        start_y = random.randint(max(0, min_end_y - crop_h), min(max_start_y, h - crop_h))
    else:
        start_x = random.randint(0, max(0, w - crop_w))
        start_y = random.randint(0, max(0, h - crop_h))
    
    # Crop image
    cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    
    # Adjust predictions
    adjusted_predictions = []
    for pred in object_predictions:
        bbox = pred.bbox.to_xyxy()
        new_bbox = [
            bbox[0] - start_x,
            bbox[1] - start_y,
            bbox[2] - start_x,
            bbox[3] - start_y,
        ]
        
        # Only keep if still within cropped region
        if new_bbox[0] >= 0 and new_bbox[1] >= 0 and new_bbox[2] <= crop_w and new_bbox[3] <= crop_h:
            pred_copy = pred.deepcopy()
            pred_copy.bbox.shift_amount = [start_x, start_y]
            adjusted_predictions.append(pred_copy)
    
    return cropped, adjusted_predictions


def create_image_pyramid(image: np.ndarray, scales: list = None, min_size: int = 32) -> list:
    """Create multi-scale image pyramid.
    
    Args:
        image: Input image
        scales: List of scale factors (default: [1.0, 0.75, 0.5, 0.25])
        min_size: Minimum dimension size for smallest scale
    
    Returns:
        List of (scale, scaled_image) tuples
    """
    if scales is None:
        scales = [1.0, 0.75, 0.5, 0.25]
    
    pyramid = []
    h, w = image.shape[:2]
    
    for scale in sorted(scales, reverse=True):
        scaled_h = int(h * scale)
        scaled_w = int(w * scale)
        
        # Skip if too small
        if scaled_h < min_size or scaled_w < min_size:
            continue
        
        scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        pyramid.append((scale, scaled_image))
    
    return pyramid


def apply_mosaic_augmentation(images: list, object_predictions_list: list, output_size: tuple = (640, 640)) -> tuple:
    """Create mosaic augmentation from 4 images.
    
    Args:
        images: List of 4 input images
        object_predictions_list: List of 4 object prediction lists
        output_size: Output image size (height, width)
    
    Returns:
        Tuple of (mosaic_image, combined_predictions)
    """
    if len(images) != 4 or len(object_predictions_list) != 4:
        raise ValueError("Mosaic augmentation requires exactly 4 images")
    
    h, w = output_size
    center_x = w // 2
    center_y = h // 2
    
    # Create blank canvas
    mosaic = np.zeros((h, w, 3), dtype=np.uint8)
    combined_predictions = []
    
    # Place images in 4 quadrants
    quadrants = [
        (0, 0, center_x, center_y),  # Top-left
        (center_x, 0, w, center_y),  # Top-right
        (0, center_y, center_x, h),  # Bottom-left
        (center_x, center_y, w, h),  # Bottom-right
    ]
    
    for idx, (img, preds, (x1, y1, x2, y2)) in enumerate(zip(images, object_predictions_list, quadrants)):
        # Resize image to fit quadrant
        quad_w = x2 - x1
        quad_h = y2 - y1
        resized = cv2.resize(img, (quad_w, quad_h))
        
        # Place in mosaic
        mosaic[y1:y2, x1:x2] = resized
        
        # Adjust predictions
        scale_x = quad_w / img.shape[1]
        scale_y = quad_h / img.shape[0]
        
        for pred in preds:
            bbox = pred.bbox.to_xyxy()
            new_bbox = [
                bbox[0] * scale_x + x1,
                bbox[1] * scale_y + y1,
                bbox[2] * scale_x + x1,
                bbox[3] * scale_y + y1,
            ]
            
            pred_copy = pred.deepcopy()
            pred_copy.bbox.shift_amount = [x1, y1]
            combined_predictions.append(pred_copy)
    
    return mosaic, combined_predictions


def apply_mixup(image1: np.ndarray, image2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Apply mixup augmentation between two images.
    
    Args:
        image1: First input image
        image2: Second input image
        alpha: Mixing ratio (0.0-1.0)
    
    Returns:
        Mixed image
    """
    # Ensure images are same size
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    # Mix images
    mixed = cv2.addWeighted(image1, alpha, image2, 1.0 - alpha, 0)
    
    return mixed


def calculate_histogram(image: np.ndarray, bins: int = 256) -> dict:
    """Calculate color histogram for image.
    
    Args:
        image: Input image
        bins: Number of bins for histogram
    
    Returns:
        Dictionary with histogram data for each channel
    """
    histograms = {}
    
    if len(image.shape) == 2:
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        histograms["gray"] = hist.flatten().tolist()
    else:
        colors = ["b", "g", "r"]
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            histograms[color] = hist.flatten().tolist()
    
    return histograms


def match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match histogram of source image to reference image.
    
    Args:
        source: Source image to transform
        reference: Reference image with target histogram
    
    Returns:
        Source image with matched histogram
    """
    # Convert to LAB color space for better results
    if len(source.shape) == 3:
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        
        # Match each channel
        matched_lab = np.zeros_like(source_lab)
        for i in range(3):
            source_hist, bins = np.histogram(source_lab[:, :, i].flatten(), 256, [0, 256])
            reference_hist, _ = np.histogram(reference_lab[:, :, i].flatten(), 256, [0, 256])
            
            source_cdf = source_hist.cumsum()
            source_cdf = source_cdf / source_cdf[-1]
            
            reference_cdf = reference_hist.cumsum()
            reference_cdf = reference_cdf / reference_cdf[-1]
            
            # Create lookup table
            lookup_table = np.zeros(256, dtype=np.uint8)
            j = 0
            for k in range(256):
                while j < 255 and source_cdf[k] > reference_cdf[j]:
                    j += 1
                lookup_table[k] = j
            
            matched_lab[:, :, i] = cv2.LUT(source_lab[:, :, i], lookup_table)
        
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale
        source_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
        reference_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
        
        source_cdf = source_hist.cumsum()
        source_cdf = source_cdf / source_cdf[-1]
        
        reference_cdf = reference_hist.cumsum()
        reference_cdf = reference_cdf / reference_cdf[-1]
        
        lookup_table = np.zeros(256, dtype=np.uint8)
        j = 0
        for k in range(256):
            while j < 255 and source_cdf[k] > reference_cdf[j]:
                j += 1
            lookup_table[k] = j
        
        return cv2.LUT(source, lookup_table)
