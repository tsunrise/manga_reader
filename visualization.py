import matplotlib.pyplot as plt
import matplotlib.patches as patches

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