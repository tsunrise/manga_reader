from dataset import MangaDataset
from model.detr import DetrFrameDetectionModel
from model.model import TrainConfig
from simple_parsing import ArgumentParser

def main():
    """
    train frame detection model using DETR
    """
    parser = ArgumentParser()
    # training config
    parser.add_arguments(TrainConfig, dest="config")
    parser.add_argument("--detection_task", type=str, help='frame detection, text detection, or frame and text detection',
                        choices=('frame', 'text', "frame_text"), default="frame")
    parser.add_argument("--exclude_book", action="append", help='exclude book from training', default=[])
    args = parser.parse_args()
    train_config: TrainConfig = args.config
    detr_model = DetrFrameDetectionModel()

    frame_annotations = args.detection_task in ("frame", "frame_text")
    text_annotations = args.detection_task in ("text", "frame_text")
    ds = MangaDataset(detr_model.image_process, frame_annotations=frame_annotations, text_annotations=text_annotations, exclude_books=args.exclude_book)

    detr_model.train(train_config, ds)

    # prediction demo
    # indices = [2,3]
    # images = [ds[i]["image"] for i in indices]
    # target_sizes = [(image.height, image.width) for image in images]
    # bbs = detr_model.predict(images, target_sizes)
    # print(bbs)

if __name__ == "__main__":
    main()