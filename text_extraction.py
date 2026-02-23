import logging
from copy import deepcopy
import numpy as np
import cv2
import easyocr
import preprocessing as preprocess

logger = logging.getLogger(__name__)


def extract_text(image_path: str, reader: easyocr.Reader) -> tuple[list, np.ndarray]:
    """
    Wrapper method handling the text extraction logic.
    :param image_path: Image path
    :param reader: Persistent EasyOCR reader object
    :return: (text_info, final_image_with_bboxes)
    """
    logger.debug("Initiating text extraction")
    extractor = TextExtraction(img_path=image_path, reader=reader)

    # Run the OCR logic
    text_info = extractor.read_text_from_image()

    # Generate the visualization
    annotated_image = extractor.draw_bounding_box(text_info)

    logger.debug("Completed text extraction")
    return text_info, annotated_image


class TextExtraction:
    # Supported language codes for EasyOCR
    LANGUAGE_CODES = [
        "abq", "ady", "af", "ang", "ar", "as", "ava", "az", "be", "bg", "bh", "bho",
        "bn", "bs", "ch_sim", "ch_tra", "che", "cs", "cy", "da", "dar", "de", "en", "es",
        "et", "fa", "fr", "ga", "gom", "hi", "hr", "hu", "id", "inh", "is", "it", "ja",
        "kbd", "kn", "ko", "ku", "la", "lbe", "lez", "lt", "lv", "mah", "mai", "mi", "mn",
        "mr", "ms", "mt", "ne", "new", "nl", "no", "oc", "pi", "pl", "pt", "ro", "ru",
        "rs_cyrillic", "rs_latin", "sck", "sk", "sl", "sq", "sv", "sw", "ta", "tab",
        "te", "th", "tjk", "tl", "tr", "ug", "uk", "ur", "uz", "vi"
    ]

    def __init__(self, img_path: str, reader: easyocr.Reader):
        """
        :param img_path: Path to file
        :param reader: The shared EasyOCR instance
        """
        # Utilizes your custom preprocessing logic (float64, RGB, alpha-blending)
        self.image: np.ndarray = preprocess.read_image(img_path)
        self.reader = reader

    def read_text_from_image(self, scale_xy: tuple[float, float] = (2.0, 2.0)) -> list:
        """
        Preprocesses image and runs OCR inference.
        """
        # 1. Store original dimensions for coordinate re-mapping
        original_h, original_w = self.image.shape[:2]
        target_w = int(original_w * scale_xy[0])
        target_h = int(original_h * scale_xy[1])

        # 2. Deep Pre-processing Pipeline
        img_work = deepcopy(self.image)
        img_work = preprocess.denoise_image(img_work)
        img_work = preprocess.scale_image(img_work, target_w, target_h)
        #img_work = preprocess.clahe_color_amplification(img_work, 0.3)


        img_for_ocr = preprocess.prepare_for_ocr(img_work)

        # 4. Contextual Inference using Word Beam Search
        extracted_text = self.reader.readtext(
            img_for_ocr,
            decoder='wordbeamsearch',
            beamWidth=5,
            rotation_info=[90, 270],
            width_ths=1,
            height_ths=0.5,
        )

        # 5. Inverse Coordinate Mapping
        # We must map coordinates from the upscaled space back to original image space
        scaled_extracted_text = []
        ratio_x = img_for_ocr.shape[1] / original_w
        ratio_y = img_for_ocr.shape[0] / original_h

        for (coords, text, prob) in extracted_text:
            # coords: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            rescaled_coords = []
            for [x, y] in coords:
                rescaled_x = int(x / ratio_x)
                rescaled_y = int(y / ratio_y)
                rescaled_coords.append([rescaled_x, rescaled_y])

            scaled_extracted_text.append((rescaled_coords, text, prob))

        return scaled_extracted_text

    def draw_bounding_box(self, scaled_extracted_text: list) -> np.ndarray:
        """
        Converts the internal float64 image to uint8 BGR and draws OCR results.
        """
        # Convert float64 [0,1] RGB to uint8 [0,255] BGR for OpenCV
        image_with_boxes = (self.image * 255).astype(np.uint8)
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)

        for bbox, text, conf in scaled_extracted_text:
            # Map coordinates to numpy array for polylines
            pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))

            # Draw Box (Red)
            cv2.polylines(image_with_boxes, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

            # Labeling logic
            text_x, text_y = bbox[0][0], bbox[0][1] - 10
            if text_y < 10: text_y = bbox[0][1] + 20

            label = f"{text} ({conf:.2f})"
            cv2.putText(image_with_boxes, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return image_with_boxes

    def remove_text_from_image(self, text_info: list) -> np.ndarray:
        """Placeholder for future Inpainting logic."""
        return deepcopy(self.image)