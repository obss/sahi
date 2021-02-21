# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import copy
import os
import random
import time

import cv2
import numpy as np
import skimage.io
from sahi.utils.file import create_dir


def crop_object_predictions(
    image: np.array,
    object_prediction_list,
    output_dir: str = "",
    file_name: str = "prediction_visual",
    export_format: str = "png",
):
    """
    Crops bounding boxes over the source image and exports it to output folder.
    Arguments:
        object_predictions: a list of prediction.ObjectPrediction
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """
    # create output folder if not present
    create_dir(output_dir)
    # add bbox and mask to image if present
    for ind, object_prediction in enumerate(object_prediction_list):
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()
        bbox = object_prediction.bbox.to_voc_bbox()
        category_id = object_prediction.category.id
        # crop detections
        # deepcopy crops so that original is not altered
        cropped_img = copy.deepcopy(
            image[
                int(bbox[1]) : int(bbox[3]),
                int(bbox[0]) : int(bbox[2]),
                :,
            ]
        )
        save_path = os.path.join(
            output_dir,
            file_name
            + "_box"
            + str(ind)
            + "_class"
            + str(category_id)
            + "."
            + export_format,
        )
        cv2.imwrite(save_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))


def convert_image_to(read_path, extension="jpg", grayscale=False):
    """
    Reads image from path and saves as given extension.
    """
    image = cv2.imread(read_path)
    pre, ext = os.path.splitext(read_path)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pre = pre + "_gray"
    save_path = pre + "." + extension
    cv2.imwrite(save_path, image)


def read_large_image(image_path):
    use_cv2 = True
    # read image, cv2 fails on large files
    try:
        # convert to rgb (cv2 reads in bgr)
        img_cv2 = cv2.imread(image_path, 1)
        image0 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    except:
        image0 = skimage.io.imread(image_path, as_grey=False).astype(np.uint8)  # [::-1]
        use_cv2 = False
    return image0, use_cv2


def read_image(image_path):
    """
    Loads image as numpy array from given path.
    """
    # read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return image
    return image


def select_random_color():
    """
    Selects random color.
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


def apply_color_mask(image: np.array, color: tuple):
    """
    Applies color mask to given input image.
    """
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    (r[image == 1], g[image == 1], b[image == 1]) = color
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask


def visualize_prediction(
    image: np.array,
    boxes,
    classes,
    masks=None,
    rect_th: float = 3,
    text_size: float = 3,
    text_th: float = 3,
    color: tuple = (0, 0, 0),
    output_dir: str = "",
    file_name: str = "prediction_visual",
):
    """
    Visualizes prediction classes, bounding boxes over the source image
    and exports it to output folder.
    """
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select random color if not specified
    if color == (0, 0, 0):
        color = select_random_color()
    # add bbox and mask to image if present
    for i in range(len(boxes)):
        # deepcopy boxso that original is not altered
        box = copy.deepcopy(boxes[i])
        class_ = classes[i]
        # visualize masks if present
        if masks is not None:
            # deepcopy mask so that original is not altered
            mask = copy.deepcopy(masks[i])
            # draw mask
            rgb_mask = apply_color_mask(np.squeeze(mask), color)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.7, 0)
        # visualize boxes
        cv2.rectangle(
            image,
            tuple(box[0:2]),
            tuple(box[2:4]),
            color=color,
            thickness=rect_th,
        )
        # arange bounding box text location
        if box[1] - 10 > 10:
            box[1] -= 10
        else:
            box[1] += 10
        # add bounding box text
        cv2.putText(
            image,
            class_,
            tuple(box[0:2]),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            color,
            thickness=text_th,
        )
    if output_dir != "":
        # create output folder if not present
        create_dir(output_dir)
        # save inference result
        save_path = os.path.join(output_dir, file_name + ".png")
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elapsed_time = time.time() - elapsed_time
    return {"image": image, "elapsed_time": elapsed_time}


def visualize_object_predictions(
    image: np.array,
    object_prediction_list,
    rect_th: float = 1,
    text_size: float = 0.3,
    text_th: float = 1,
    color: tuple = (0, 0, 0),
    output_dir: str = "",
    file_name: str = "prediction_visual",
    export_format: str = "png",
):
    """
    Visualizes prediction category names, bounding boxes over the source image
    and exports it to output folder.
    Arguments:
        object_prediction_list: a list of prediction.ObjectPrediction
        rect_th: rectangle thickness
        text_size: size of the category name over box
        text_th: text thickness
        color: annotation color in the form: (0, 255, 0)
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select random color if not specified
    if color == (0, 0, 0):
        color = select_random_color()
    # add bbox and mask to image if present
    for object_prediction in object_prediction_list:
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()

        bbox = object_prediction.bbox.to_voc_bbox()
        category_name = object_prediction.category.name
        score = object_prediction.score.score

        # visualize masks if present
        if object_prediction.mask is not None:
            # deepcopy mask so that original is not altered
            mask = object_prediction.mask.bool_mask
            # draw mask
            rgb_mask = apply_color_mask(mask, color)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.4, 0)
        # visualize boxes
        cv2.rectangle(
            image,
            tuple(bbox[0:2]),
            tuple(bbox[2:4]),
            color=color,
            thickness=rect_th,
        )
        # arange bounding box text location
        if bbox[1] - 5 > 5:
            bbox[1] -= 5
        else:
            bbox[1] += 5
        # add bounding box text
        label = "%s %.2f" % (category_name, score)
        cv2.putText(
            image,
            label,
            tuple(bbox[0:2]),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            color,
            thickness=text_th,
        )
    if output_dir != "":
        # create output folder if not present
        create_dir(output_dir)
        # save inference result
        save_path = os.path.join(output_dir, file_name + "." + export_format)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elapsed_time = time.time() - elapsed_time
    return {"image": image, "elapsed_time": elapsed_time}


def get_coco_segmentation_from_bool_mask(bool_mask):
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
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    polygons = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1)
    )
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    # Convert polygon to coco segmentation
    coco_segmentation = [polygon.flatten().tolist() for polygon in polygons]
    return coco_segmentation


def get_bool_mask_from_coco_segmentation(coco_segmentation, width, height):
    """
    Convert coco segmentation to 2D boolean mask of given height and width
    """
    size = [height, width]
    points = [
        np.array(point).reshape(-1, 2).round().astype(int)
        for point in coco_segmentation
    ]
    bool_mask = np.zeros(size)
    bool_mask = cv2.fillPoly(bool_mask, points, 1)
    bool_mask.astype(np.bool)
    return bool_mask


def get_bbox_from_bool_mask(bool_mask):
    """
    Generate voc bbox ([xmin, ymin, xmax, ymax]) from given bool_mask (2D np.ndarray)
    """
    rows = np.any(bool_mask, axis=1)
    cols = np.any(bool_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return []

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    # width = xmax - xmin
    # height = ymax - ymin

    return [xmin, ymin, xmax, ymax]


def normalize_numpy_image(image: np.ndarray):
    """
    Normalizes numpy image
    """
    return image / np.max(image)


def ipython_display(image):
    """
    Displays numpy image in notebook.

    If input image is in range 0..1, please first multiply img by 255
    Assumes image is ndarray of shape [height, width, channels] where channels can be 1, 3 or 4
    """
    import IPython

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, ret = cv2.imencode(".png", image)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)
