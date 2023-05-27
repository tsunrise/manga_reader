from typing import Optional
from PIL.Image import Image
from dataset import COCOTarget, ImageProcessor
from manga109utils import BoundingBox
from model.model import DetectionModel

from transformers import DetrImageProcessor, PreTrainedModel, DetrForObjectDetection
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput
from constant import FRAME_LABEL, TEXT_LABEL

_category_id_to_label = {
    FRAME_LABEL: "frame",
    TEXT_LABEL: "text",
}

class DetrImagePipeline(ImageProcessor):
    def __init__(self, pretrained_model_name_or_path: Optional[str] = None) -> None:
        super().__init__()
        if pretrained_model_name_or_path is not None:
            self.processor = DetrImageProcessor.from_pretrained(pretrained_model_name_or_path)
        else:
            self.processor = DetrImageProcessor()


    def preprocess(self, images: list[Image], targets: Optional[list[COCOTarget]] = None) -> dict:
        return self.processor.preprocess(images, targets, return_tensors="pt")
    
    def post_process_to_bounding_box_list(self, output: DetrObjectDetectionOutput, target_sizes: list[tuple]) -> list[list[tuple[BoundingBox, float]]]:
        # NOTE: this method modifies the output object
        output.logits = output.logits.cpu()
        output.pred_boxes = output.pred_boxes.cpu()
        out = self.processor.post_process_object_detection(output, threshold=0.5, target_sizes=target_sizes)
        result = []
        for item in out:
            item_result = []
            boxes = item["boxes"]
            scores = item["scores"]
            labels = item["labels"]
            for box, score, label in zip(boxes, scores, labels):
                xmin, ymin, xmax, ymax = box
                item_result.append((BoundingBox(float(xmin), float(ymin), float(xmax), float(ymax), bbtype=label), score))
            result.append(item_result)
        return result
    
class DetrFrameDetectionModel(DetectionModel):
    def _default_pretrained_name_or_path(self) -> str:
        return "facebook/detr-resnet-50"

    def build_model(self, name_or_path) -> PreTrainedModel:
        return DetrForObjectDetection.from_pretrained(name_or_path)
    
    def build_image_processor(self, _name_or_path) -> ImageProcessor:
        return DetrImagePipeline(self._default_pretrained_name_or_path())
    

        


        

