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
    args = parser.parse_args()
    train_config: TrainConfig = args.config
    detr_model = DetrFrameDetectionModel()

    ds = MangaDataset(detr_model.image_process, frame_annotations=True, text_annotations=False)

    detr_model.train(train_config, ds)

    # prediction demo
    # indices = [2,3]
    # images = [ds[i]["image"] for i in indices]
    # target_sizes = [(image.height, image.width) for image in images]
    # bbs = detr_model.predict(images, target_sizes)
    # print(bbs)

if __name__ == "__main__":
    main()