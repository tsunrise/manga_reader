import matplotlib.pyplot as plt
import matplotlib.patches as patches
from retrieve import EndToEndSceneRetriever
from PIL import Image

def visualize_scene_retrieval(scene_retriever, book, test_queries, test_expects, top_k=5, offset=0.2):
  pages = list(book.get_page_iter())
  for query, expect in zip(test_queries, test_expects):
    print(f"query: {query}")
    result = scene_retriever.query_plus(query, top_k=5)
    fig, axs = plt.subplots(1, top_k, figsize=(20, 5))
    for i, item in enumerate(result):
      image = pages[item[0]["page"]].get_image()
      bbox = item[0]["location"]
      x_offset, y_offset = offset*(bbox["xmax"]-bbox["xmin"]), offset*(bbox["ymax"]-bbox["ymin"])
      cropped_image = image.crop((bbox["xmin"]-x_offset, bbox["ymin"]-y_offset, bbox["xmax"]+x_offset, bbox["ymax"]+y_offset))
      axs[i].imshow(cropped_image)
      axs[i].axis('off')
      axs[i].add_patch(
        patches.Rectangle(xy=(x_offset, y_offset),
                          width=bbox["xmax"]-bbox["xmin"],
                          height=bbox["ymax"]-bbox["ymin"],
                          linewidth=3,
                          linestyle="solid",
                          ec="red",
                          fill=False))
      axs[i].set_title("correct page" if item[0]["page"]==expect else "incorrect page")
    plt.show()


def visualize_scene_retrieval_novel_book(scene_retriever: EndToEndSceneRetriever, images: list[Image.Image], test_queries, top_k=5, offset=0.2):
  for query in test_queries:
    print(f"query: {query}")
    result = scene_retriever.query_plus(query, top_k=5)
    fig, axs = plt.subplots(1, top_k, figsize=(20, 5))
    for i, item in enumerate(result):
      image = images[item[0]["page"]]
      bbox = item[0]["location"]
      x_offset, y_offset = offset*(bbox["xmax"]-bbox["xmin"]), offset*(bbox["ymax"]-bbox["ymin"])
      cropped_image = image.crop((bbox["xmin"]-x_offset, bbox["ymin"]-y_offset, bbox["xmax"]+x_offset, bbox["ymax"]+y_offset))
      axs[i].imshow(cropped_image)
      axs[i].axis('off')
      axs[i].add_patch(
        patches.Rectangle(xy=(x_offset, y_offset),
                          width=bbox["xmax"]-bbox["xmin"],
                          height=bbox["ymax"]-bbox["ymin"],
                          linewidth=3,
                          linestyle="solid",
                          ec="red",
                          fill=False))
    plt.show()

def filename_to_int(filename: str):
  # remove suffix (note that filename can contain multiple dots, and we only want to remove the last one)
  filename = filename.rsplit(".", 1)[0]
  # keep only numbers
  filename = "0" + "".join([c for c in filename if c.isdigit()])
  return int(filename)

def load_images_from_folder(folder):
  import os
  images = {}
  for filename in os.listdir(folder):
    try:
      images[filename] = Image.open(os.path.join(folder,filename))
    except:
      pass
    # sort filenames using intuitive order
  return [images[filename] for filename in sorted(images.keys(), key=filename_to_int)]
  