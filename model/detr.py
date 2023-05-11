from typing import Optional
from PIL.Image import Image
from dataset import COCOTarget, ImageProcessor
from manga109utils import BoundingBox
from model.model import DetectionModel

from transformers import DetrImageProcessor, PreTrainedModel, DetrForObjectDetection

class DetrImagePipeline(ImageProcessor):
    def __init__(self, pretrained_model_name_or_path: Optional[str] = None) -> None:
        super().__init__()
        if pretrained_model_name_or_path is not None:
            self.processor = DetrImageProcessor.from_pretrained(pretrained_model_name_or_path)
        else:
            self.processor = DetrImageProcessor()


    def preprocess(self, images: list[Image], targets: Optional[list[COCOTarget]] = None) -> dict:
        return self.processor.preprocess(images, targets, return_tensors="pt")
    
    def post_process_to_bounding_box_list(self, output, target_sizes: list[tuple]) -> list[list[tuple[BoundingBox, float]]]:
        out = self.processor.post_process_object_detection(output, target_sizes)
        result = []
        for item in out:
            boxes = item["boxes"]
            scores = item["scores"]
            for box, score in zip(boxes, scores):
                xmin, ymin, xmax, ymax = box
                result.append((BoundingBox(xmin, xmax, ymin, ymax), score))
        return result
    
class DetrFrameDetectionModel(DetectionModel):
    def build_model(self) -> PreTrainedModel:
        return DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    def build_image_processor(self) -> ImageProcessor:
        return DetrImagePipeline("facebook/detr-resnet-50")
    

        


        

