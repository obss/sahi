import numpy as np
from sahi import DetectionModel
from typing import Any
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
import os
import cv2
from transformers import Mask2FormerForUniversalSegmentation, MaskFormerForInstanceSegmentation, OneFormerForUniversalSegmentation, BaseImageProcessor
from enum import Enum
import torch
from transformers import AutoModelForUniversalSegmentation, Mask2FormerImageProcessor, MaskFormerImageProcessor, OneFormerProcessor


class SegmentationType(Enum):
    INSTANCE_SEGMENTATION = "instance"
    SEMANTIC_SEGMENTATION = "semantic"
    PANOPTIC_SEGMENTATION = "panoptic"


class HuggingFaceUniversalSegmentationModel(DetectionModel):
    
       
    """
        Currently We have only support for the following supported_models_and_processors.

        Args :
            overlap_mask_area_threshold(`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            label_ids_to_fuse(`Set[int]`, *optional*):
                The labels in this state will have all their instances be fused together. For instance we could say
                there can only be one sky in an image, but several persons, so the label ID for sky would be in that
                set, but not the one for person.
             min_segment_area(`int`,*optional*, defaults to 100)
                If the area of a segment is less than the provided min_segment_area, then the segment is ignored.
             segmentation_type(`segmentation_type enum`, *optional*, defaults to INSTANCE_SEGMENTATION)
                The segmentation type the model uses.
            Not all params are valid for all segmentation types, so if not valid they are simple ignored.
    """
    supported_models_and_processors = {
            Mask2FormerForUniversalSegmentation : Mask2FormerImageProcessor,
            MaskFormerForInstanceSegmentation : MaskFormerImageProcessor,
            OneFormerForUniversalSegmentation : OneFormerProcessor
        }
    
    
    def __init__(
        self,
        model_path: str | None = None,
        model: Any | None = None,
        processor: Any | None = None,
        config_path: str | None = None,
        device: str | None = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        category_mapping: dict | None = None,
        category_remapping: dict | None = None,
        load_at_init: bool = True,
        image_size: int | None = None,
        token: str | None = None,
        overlap_mask_area_threshold : float = 0.8,
        label_ids_to_fuse : list[int] | None = None,
        min_segment_area : int = 100,
        segmentation_type : SegmentationType = SegmentationType.INSTANCE_SEGMENTATION
    ):
        self._processor = processor
        self._output_image_shapes: list = []
        self._token = token
        self.segmentation_type = segmentation_type
        self.overlap_mask_area_threshold = overlap_mask_area_threshold
        self.label_ids_to_fuse = label_ids_to_fuse
        self.min_segment_area = min_segment_area
    
        super().__init__(
            model_path,
            model,
            config_path,
            device,
            mask_threshold,
            confidence_threshold,
            category_mapping,
            category_remapping,
            load_at_init,
            image_size,
        )

        

        
    @property
    def processor(self):
        return self._processor

    @property
    def image_shapes(self):
        return self._output_image_shapes

    @property
    def num_categories(self) -> int:
        return self.model.config.num_labels

    def load_model(self):
        """Load model from HuggingFace."""
        hf_token = os.getenv("HF_TOKEN", self._token)
        assert self.model_path is not None, "model_path must be provided for HuggingFace models"
        model = AutoModelForUniversalSegmentation.from_pretrained(self.model_path, token=hf_token)

        processor_class = self.supported_models_and_processors.get(type(model), None)

        assert processor_class is not None, f'model of type {type(model)} is not supported. supported models are: {list(self.supported_models_and_processors.keys())}'

        if self.image_size is not None:

            size = {"height": self.image_size, "width": self.image_size}
           
            # use_fast=True raises error: AttributeError: 'SizeDict' object has no attribute 'keys'
            processor = processor_class.from_pretrained(self.model_path, size=size, do_resize=True, use_fast=False, token=hf_token)
            
        else:
            processor = processor_class.from_pretrained(self.model_path, use_fast=False, token=hf_token)
        
        self.set_model(model, processor) 

    def set_model(self, model: Any, processor: Any = None, **kwargs):
        processor = processor or self.processor
        if processor is None:
            raise ValueError(f"'processor' is required to be set, got {processor}.")
        
        is_valid_pair = False
        for model_type, processor_type in self.supported_models_and_processors.items():
            if isinstance(model, model_type) and isinstance(processor, processor_type):
                is_valid_pair = True
                break
        if not is_valid_pair:
            raise ValueError(f"Invalid model and processor pair: {type(model)} and {type(processor)}. Supported pairs are: {self.supported_models_and_processors}")
       
            
        self.model = model
        self.model.to(self.device) 
        self._processor = processor
        self.category_mapping = self.model.config.id2label  #

        if isinstance(self.model, MaskFormerForInstanceSegmentation) or isinstance(self.model, Mask2FormerForUniversalSegmentation):
            self.prepost_handler = MaskFormerAndMask2FormerPrePostHandler(self)
        else:
            self.prepost_handler = OneFormerPrePostHandler(self)

    def perform_inference(self, image: list | np.ndarray):

        if self.model is None or self.processor is None:
            raise RuntimeError(f'{self.model is None} {self.processor is None}')
        
        inputs = self.prepost_handler.handle_pre_process(image)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if isinstance(image, list):
            self._output_image_shapes = [(img.shape[0], img.shape[1]) for img in image]
        else:
            self._output_image_shapes = [(image.shape[0], image.shape[1])]
      
        self._original_predictions = outputs

    def get_valid_predictions(self, post_processed_output) -> tuple:
       
        scores=[]
        category_ids=[]
        #returns polygons as list of lists where inner list is flattened as [x1, y1, ...,xn,yn]
        polygonal_segments=[]
      
        segments = post_processed_output['segmentation']
        segments_info = post_processed_output['segments_info']

        if segments is None or not segments_info:
            return scores, category_ids, polygonal_segments
    
        for segment, segment_info in zip(segments, segments_info):
            mask = segment.cpu().numpy().astype(np.uint8)
            score = segment_info['score']
            category_id = segment_info['label_id']
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area >= self.min_segment_area:
                        polygonal_segments.append([contour.squeeze(1).flatten().tolist()])
                        scores.append(score)
                        category_ids.append(category_id)
                
        return scores, category_ids, polygonal_segments
    
    def perform_batch_inference(self, images: list[np.ndarray]) -> None:
        return self.perform_inference(images)
    
    

    #performs inference on the list of slices
    #shift amount [[shift_x, shift_y],...] gives how much each slice must be shifted
    #full_shape [[width, height]] gives each slices dimension
    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int | float]] | None = [[0, 0]],
        full_shape_list: list[list[int | float]] | None = None,
    ):
        
        original_predictions = self._original_predictions
        target_sizes = self._output_image_shapes

        post_processed_outputs = self.prepost_handler.handle_post_process(original_predictions, target_sizes)
        
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        n_image = len(post_processed_outputs)
        object_prediction_list_per_image = []
        for image_ind in range(n_image):
           
            scores, category_ids, segments = self.get_valid_predictions(post_processed_outputs[image_ind])

            object_prediction_list = []

            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            
            #iterate each polygonal segment
            for ind in range(len(segments)):
                category_id = category_ids[ind]
                segment = segments[ind]
                score = scores[ind]
                
                #8 represents the numer of x,y coords of a polygon
                #we need atleast 8 to have a polygon
                if len(segment[0]) >= 8:
                    object_prediction = ObjectPrediction(
                        bbox=None,
                        segmentation=segment,
                        category_id=category_id,
                        category_name=self.category_mapping[category_id],
                        shift_amount=shift_amount,
                        score=score,
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


class PrePostHandler:
    """
        A Handler Base class that provides the ability for the universal segmentation models to add their own pre and post processing ability.
    """
    
    def __init__(self, hugging_face_universal_segmentation_model : HuggingFaceUniversalSegmentationModel):
        """
            Args:
            hugging_face_universal_segmentation_model : HuggingFaceUniversalSegmentationModel,
            This is used by the pre and post processing methods to look up for the appropriate params.
            This includes params like mask_threshold, confidence_threshold etc.
        """
        self.hf_universal_seg = hugging_face_universal_segmentation_model
    

   
    def _convert_semantic_mask_to_binary_masks(self, class_masks : list[torch.Tensor]):
        """
            To have a common output format across segmention types, the segments of semantic are converted into a common format.
            This method converts each segment into a binary mask and a separate segment_info for each of the binary mask.

            Args:
            class_masks : tensor of shape [batch_size,H,W] with each value of the tensor corresponds to a label_id.
            
            Returns:
            list[dict] each entry corresponds to one output, with each dict having two keys
            'segmentation': value is a list, this includes the binary mask for each of the segments in a output.
            'segment_info': value is a list, this includes the corresponding segment info. 
        """
        outputs = []
        for class_mask in class_masks:
            output = {'segmentation':[], 'segments_info':[]}
            label_ids = torch.unique(class_mask)
            for label_id in label_ids:
                mask = (class_mask == label_id).to(torch.uint8)
                output['segmentation'].append(mask)
                output['segments_info'].append({'score':1.0, 'label_id':label_id.item()})
            
            outputs.append(output)
        return outputs
    

    def _convert_panoptic_mask_to_binary_masks(self, post_processed_outputs : list[dict]):
        """
            To have a common output format across segmention types, the segments of panoptic are converted into a common format.
            This method converts each segment into a binary mask and a separate segment_info for each of the binary mask.

            Args:
            post_processed_outputs : list[dict], each entry corresponds to one output.
            dict has two keys 
            'segmentation': tensor of [H,W], with each value of the tensor corresponds to a label_id.
            'segments_info': list[dict], each entry represents only segment's info
            
            Returns:
            list[dict] each entry corresponds to one output, with each dict having two keys
            'segmentation': value is a list, this includes the binary mask for each of the segments in a output.
            'segment_info': value is a list, this includes the corresponding segment info. 
        """
        outputs = []
        for post_processed_output in post_processed_outputs:
            segmentation = post_processed_output['segmentation']
            segments_info = post_processed_output['segments_info']
            if segmentation is None or not segments_info:
                    continue
            segments_info_map = {segment_info['id']:segment_info for segment_info in segments_info}
            
            output = {'segmentation':[], 'segments_info':[]}
            segment_ids = torch.unique(segmentation)
            for segment_id in segment_ids.tolist():
                #segment id is zero for background pixels
                if segment_id in segments_info_map:
                    mask = (segmentation == segment_id).to(torch.uint8)
                    output['segmentation'].append(mask)
                    output['segments_info'].append(segments_info_map[segment_id])
            
            outputs.append(output)
        return outputs

    def handle_pre_process(self, image:list | np.ndarray):
        """
            This method handles the pre processing logic of the model and moves the input tensors
            required by the model appropriately to the device as provided in the hugging_face_universal_segmentation_model.

            Args:
            image: List of images or image to be provided as the input to the model.

            Returns:
            inputs: list[dict], each entry corresponds to one input image.
            dict:keys are the params for the model and values are of torch.tensor.
    
        """
        raise NotImplementedError('implement handle_pre_process')
    

    def handle_post_process(self, original_predictions : Any, target_sizes : list)->list[dict]:
        """
            This method handles the post processing logic of the model
            and outputs a common format for all the segmentation type.

            Args:
            original_predictions: This must be the output of the model as is. 
            target_sizes: For each of the output, provide the target size to resize the output.

            Returns:
            a list of dict, each entry corresponds to one output, with two keys 
            'segmentation': value is a list, this includes the binary mask for each of the segments in a output.
            'segment_info': value is a list, this includes the corresponding segment info. 
        
        """
        raise NotImplementedError('implement handle_post_process')

class MaskFormerAndMask2FormerPrePostHandler(PrePostHandler):
    """
        The Maskformer and Mask2former both uses the same pre and post processing logic.
        So this class handles both the model's pre and post processing.
    """
    def __init__(self, hugging_face_universal_segmentation : HuggingFaceUniversalSegmentationModel):
        super().__init__(hugging_face_universal_segmentation_model=hugging_face_universal_segmentation)
    
    def handle_pre_process(self, image:list | np.ndarray):
        processor = self.hf_universal_seg.processor
        assert processor, 'processor is none'
        inputs = processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs.pixel_values.to(self.hf_universal_seg.device)
        if hasattr(inputs, "pixel_mask"):
            inputs["pixel_mask"] = inputs.pixel_mask.to(self.hf_universal_seg.device)
        
        return inputs
    

    def handle_post_process(self, original_predictions : Any, target_sizes : list)->list[dict]:
        
        processor = self.hf_universal_seg.processor
        universal_segmentation = self.hf_universal_seg

        assert processor, 'processor is none'

        if universal_segmentation.segmentation_type == SegmentationType.INSTANCE_SEGMENTATION:
            return processor.post_process_instance_segmentation(
                    original_predictions, threshold = self.hf_universal_seg.confidence_threshold, 
                    mask_threshold = self.hf_universal_seg.mask_threshold,
                    overlap_mask_area_threshold = self.hf_universal_seg.overlap_mask_area_threshold,
                    target_sizes = target_sizes,
                    return_binary_maps=True
                )
        elif universal_segmentation.segmentation_type == SegmentationType.SEMANTIC_SEGMENTATION:
            post_processed_outputs = processor.post_process_semantic_segmentation(original_predictions, target_sizes)
            return self._convert_semantic_mask_to_binary_masks(post_processed_outputs)
        
        post_processed_outputs = processor.post_process_panoptic_segmentation(original_predictions,
                                        threshold = self.hf_universal_seg.confidence_threshold, 
                                        mask_threshold = self.hf_universal_seg.mask_threshold,
                                        overlap_mask_area_threshold = self.hf_universal_seg.overlap_mask_area_threshold,
                                        target_sizes = target_sizes,
                                        label_ids_to_fuse = self.hf_universal_seg.label_ids_to_fuse)
        
        return self._convert_panoptic_mask_to_binary_masks(post_processed_outputs)
    

class OneFormerPrePostHandler(PrePostHandler):
    """
        Handles pre and post processing for OneFormer.
        Unlike MaskFormer/Mask2Former, OneFormer requires a task_input during 
        pre-processing.
    """
    def __init__(self, hugging_face_universal_segmentation : HuggingFaceUniversalSegmentationModel):
        super().__init__(hugging_face_universal_segmentation_model=hugging_face_universal_segmentation)

   
    def _convert_instance_mask_to_binary_masks(self, post_processed_outputs : list[dict]):
        """
          oneformer's instance segmentation post process returns masks in the same format as panoptic.
        """
        return self._convert_panoptic_mask_to_binary_masks(post_processed_outputs)


    def handle_pre_process(self, image:list | np.ndarray):
        processor = self.hf_universal_seg.processor
        segmentation_type = self.hf_universal_seg.segmentation_type
        assert processor, 'processor is none'
       
        task_inputs = [segmentation_type.value]*len(image) if isinstance(image, list) else [segmentation_type.value]
        inputs = processor(images=image, task_inputs=task_inputs, return_tensors="pt")

        inputs["pixel_values"] = inputs.pixel_values.to(self.hf_universal_seg.device)
        if hasattr(inputs, "pixel_mask"):
            inputs["pixel_mask"] = inputs.pixel_mask.to(self.hf_universal_seg.device)
        
        inputs["task_inputs"] = inputs.task_inputs.to(self.hf_universal_seg.device)
        
        return inputs

    def handle_post_process(self, original_predictions : Any, target_sizes : list)->list[dict]:
        
        processor = self.hf_universal_seg.processor
        universal_segmentation = self.hf_universal_seg

        assert processor

        if universal_segmentation.segmentation_type == SegmentationType.INSTANCE_SEGMENTATION:
            post_processed_outputs = processor.post_process_instance_segmentation(
                    original_predictions, threshold = self.hf_universal_seg.confidence_threshold, 
                    mask_threshold = self.hf_universal_seg.mask_threshold,
                    overlap_mask_area_threshold = self.hf_universal_seg.overlap_mask_area_threshold,
                    target_sizes = target_sizes
                )
            return self._convert_instance_mask_to_binary_masks(post_processed_outputs)
        elif universal_segmentation.segmentation_type == SegmentationType.SEMANTIC_SEGMENTATION:
            post_processed_outputs = processor.post_process_semantic_segmentation(original_predictions, target_sizes)
            return self._convert_semantic_mask_to_binary_masks(post_processed_outputs)
        
        post_processed_outputs = processor.post_process_panoptic_segmentation(original_predictions,
                                        threshold = self.hf_universal_seg.confidence_threshold, 
                                        mask_threshold = self.hf_universal_seg.mask_threshold,
                                        overlap_mask_area_threshold = self.hf_universal_seg.overlap_mask_area_threshold,
                                        target_sizes = target_sizes,
                                        label_ids_to_fuse = self.hf_universal_seg.label_ids_to_fuse)
        
        return self._convert_panoptic_mask_to_binary_masks(post_processed_outputs)