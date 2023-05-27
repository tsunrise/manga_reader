DETR_PRETRAINED = 'facebook/detr-resnet-50'
MANGA109_ROOT = 'datasets/Manga109_released_2021_12_30'

# TODO: DETR uses 0 for N/A, but we use 0 for frame, which may not be a good idea
# we may want to change this to 1 and 2, and fine-tune again
FRAME_LABEL = 0
TEXT_LABEL = 1