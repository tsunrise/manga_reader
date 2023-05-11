import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from manga_ocr import MangaOcr

def get_ocr_page(page, mocr=None):
	if not mocr:
		mocr = MangaOcr()

	image = page["image"]
	predicted_text = mocr(image)
	return predicted_text

def get_groundtruth_bb_ocr_page(page, mocr=None):
	if not mocr:
		mocr = MangaOcr()

	image = np.asarray(page["image"])
	page_predict_text = []
	for bb in page["target"]["annotations"]:
		if bb["category_id"] == 1: # text bounding box
			text_bb = bb["bbox"]
			x_min, y_min = text_bb[0], text_bb[1]
			w, h = text_bb[2], text_bb[3]
			x_max, y_max = x_min + w, y_min + h
			text_bb_array = image[int(y_min): int(y_max),
						   int(x_min): int(x_max),
						   :]
			text_bb_image = Image.fromarray(text_bb_array)
			predict_text = mocr(text_bb_image)
			page_predict_text.append(predict_text)
	return ' '.join(page_predict_text)

def get_ocr_dataset(dataset, mocr=None, method="entire_page"):
	all_page_transcript = []
	for page in tqdm(dataset, desc="Loading Pages", total=len(dataset)):
		if method == "entire_page":
			page_transcript = get_ocr_page(page, mocr)
		elif method == "ground_truth_bb":
			page_transcript = get_groundtruth_bb_ocr_page(page, mocr)
		else:
			raise NotImplementedError
		all_page_transcript.append(page_transcript)
	return all_page_transcript

def display_groundtruth_bb_ocr_page(page, mocr=None):
	if not mocr:
		mocr = MangaOcr()
	
	image = np.asarray(page["image"])
	for bb in page["target"]["annotations"]:
		if bb["category_id"] == 1: # text bounding box
			text_bb = bb["bbox"]
			plt.figure(figsize=(4, 3))
			ax = plt.gca()
			text_bb = bb["bbox"]
			x_min, y_min = text_bb[0], text_bb[1]
			w, h = text_bb[2], text_bb[3]
			x_max, y_max = x_min + w, y_min + h
			text_bb_array = image[int(y_min): int(y_max),
						   int(x_min): int(x_max),
						   :]
			text_bb_image = Image.fromarray(text_bb_array)
			predicted_text = mocr(text_bb_image)

			ax.imshow(text_bb_array)
			print(f"predicted_text: {predicted_text}")
			plt.show()
