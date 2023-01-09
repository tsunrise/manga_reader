from transformers import DetrForObjectDetection

from constant import DETR_PRETRAINED

def get_model() -> DetrForObjectDetection:
    return DetrForObjectDetection.from_pretrained(DETR_PRETRAINED, num_labels=1, ignore_mismatched_sizes=True) # type: ignore