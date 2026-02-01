import os
import cv2
import numpy as np
import easyocr
from pathlib import Path
import json
import shutil

LANGUAGE__CODES_HASHMAP = {
    "Abaza": "abq", "Adyghe": "ady", "Afrikaans": "af", "Angika": "ang", "Arabic": "ar", "Assamese": "as",
    "Avar": "ava", "Azerbaijani": "az", "Belarusian": "be", "Bulgarian": "bg", "Bihari": "bh", "Bhojpuri": "bho",
    "Bengali": "bn", "Bosnian": "bs", "Simplified Chinese": "ch_sim", "Traditional Chinese": "ch_tra", "Chechen": "che",
    "Czech": "cs", "Welsh": "cy", "Danish": "da", "Dargwa": "dar", "German": "de", "English": "en", "Spanish": "es",
    "Estonian": "et", "Persian (Farsi)": "fa", "French": "fr", "Irish": "ga", "Goan Konkani": "gom", "Hindi": "hi",
    "Croatian": "hr", "Hungarian": "hu", "Indonesian": "id", "Ingush": "inh", "Icelandic": "is", "Italian": "it",
    "Japanese": "ja", "Kabardian": "kbd", "Kannada": "kn", "Korean": "ko", "Kurdish": "ku", "Latin": "la",
    "Lak": "lbe", "Lezghian": "lez", "Lithuanian": "lt", "Latvian": "lv", "Magahi": "mah", "Maithili": "mai",
    "Maori": "mi", "Mongolian": "mn", "Marathi": "mr", "Malay": "ms", "Maltese": "mt", "Nepali": "ne", "Newari": "new",
    "Dutch": "nl", "Norwegian": "no", "Occitan": "oc", "Pali": "pi", "Polish": "pl", "Portuguese": "pt",
    "Romanian": "ro", "Russian": "ru", "Serbian (cyrillic)": "rs_cyrillic", "Serbian (latin)": "rs_latin",
    "Nagpuri": "sck", "Slovak": "sk", "Slovenian": "sl", "Albanian": "sq", "Swedish": "sv", "Swahili": "sw",
    "Tamil": "ta", "Tabassaran": "tab", "Telugu": "te", "Thai": "th", "Tajik": "tjk", "Tagalog": "tl", "Turkish": "tr",
    "Uyghur": "ug", "Ukranian": "uk", "Urdu": "ur", "Uzbek": "uz", "Vietnamese": "vi"
}

LANGUAGE_CODES = [
    "abq", "ady", "af", "ang", "ar", "as", "ava", "az", "be", "bg", "bh", "bho",
    "bn", "bs", "ch_sim", "ch_tra", "che", "cs", "cy", "da", "dar", "de", "en", "es",
    "et", "fa", "fr", "ga", "gom", "hi", "hr", "hu", "id", "inh", "is", "it", "ja",
    "kbd", "kn", "ko", "ku", "la", "lbe", "lez", "lt", "lv", "mah", "mai", "mi", "mn",
    "mr", "ms", "mt", "ne", "new", "nl", "no", "oc", "pi", "pl", "pt", "ro", "ru",
    "rs_cyrillic", "rs_latin", "sck", "sk", "sl", "sq", "sv", "sw", "ta", "tab",
    "te", "th", "tjk", "tl", "tr", "ug", "uk", "ur", "uz", "vi"
]

def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # If it's a 4-channel image (RGBA)
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Separate alpha and BGR
        alpha = img[:, :, 3] / 255.0
        bgr = img[:, :, :3]
        # Create white background
        white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
        # Blend: (Foreground * Alpha) + (Background * (1 - Alpha))
        img = (bgr * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)

    return upscale_image(img)

def upscale_image(img, scale_factor=2.0):
    """
    Upscales the image using Lanczos-4 interpolation to preserve text sharpness.
    :param img: The BGR image from read_image.
    :param scale_factor: How much to enlarge (2.0 is usually the 'sweet spot').
    :return: The upscaled image.
    """
    if img is None:
        return None

    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    new_size = (width, height)

    # INTER_LANCZOS4 is slower but prevents the 'staircase' effect on map lines
    upscaled = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)

    return upscaled

def check_language_code(langages: list[str]):
    """
    Check if the provided language codes are valid.
    :param langages: List of language codes to check.
    :return: None
    """
    for code in langages:
        if code not in LANGUAGE_CODES:
            raise ValueError(f"Invalid language code: {code}")

def save_benchmark(output_path: str, results: list):

    benchmark_data = []
    for (bbox, text, conf) in results:
        # We structure this as a list of dictionaries for easy access in tests
        benchmark_data.append({
            "text": text,
            "confidence": float(conf),
            "bbox": bbox
        })

    # ensure_ascii=False allows French characters like 'é' to be saved correctly
    # indent=4 makes the file human-readable
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=4)

    txt_output_path = Path(output_path).with_suffix('.txt')

    with open(txt_output_path, 'w', encoding='utf-8') as f:
        for (bbox, text, conf) in results:
            f.write(f"Read word: {text}\n")
            f.write(f"its position: {bbox}\n")
            f.write(f"confidence: {conf:.4f}\n")  # formatted to 4 decimal places
            f.write("-\n")

def erase_text(full_input_path, results, output_path):

    img = read_image(full_input_path)
    if img is None: return

    # Create a mask for all text found
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for (bbox, text, prob) in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    result_img = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    cv2.imwrite(output_path, result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])

def fastnl_denoising(image_path, output_path):

    img = read_image(image_path)
    if img is None:
        return None

    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 4, 5, 7, 21)

    cv2.imwrite(output_path, denoised_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return denoised_img

def bilateral_denoising(image_path, output_path):

    img = read_image(image_path)
    if img is None: return None

    # d: Diameter of pixel neighborhood
    # sigmaColor: Small value (20-30) ensures dark tones stay distinct
    # sigmaSpace: Higher value (75) allows smoothing over larger paper areas
    processed = cv2.bilateralFilter(img, d=9, sigmaColor=25, sigmaSpace=75)

    cv2.imwrite(output_path, processed)
    return processed

def shift_denoising(image_path, output_path):
    img = read_image(image_path)
    # sp = spatial window radius (higher = more smoothing of shapes)
    # sr = color window radius (higher = more colors merged together)
    cleaned = cv2.pyrMeanShiftFiltering(img, sp=10, sr=30)
    cv2.imwrite(output_path, cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return cleaned

def clahe_sharpening(img, output_path):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    merged_lab = cv2.merge((cl, a_channel, b_channel))
    final_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, final_img)
    return final_img

def gamma_sharpening(img, output_path):
    gamma = 0.6
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    final_image = cv2.LUT(img, table)
    cv2.imwrite(output_path, final_image)
    return final_image

def sauvola_sharpening(img, output_path):
    window_size = 25
    k = 0.2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean = cv2.blur(gray, (window_size, window_size))
    sq_mean = cv2.blur(gray ** 2, (window_size, window_size))
    std = np.sqrt(sq_mean - mean ** 2)

    # Formule de Sauvola : T = m * (1 + k * (std / R - 1))
    r1 = 128  # Valeur standard pour les images 8-bits
    threshold = mean * (1 + k * (std / r1 - 1))

    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    cv2.imwrite(output_path, binary)
    return binary

def lcn_sharpening(img, output_path):
    window_size = 15

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # 1. On sépare d'abord (renvoie un tuple de 3 arrays)
    l, a, b = cv2.split(lab)

    # 2. On convertit ensuite le canal L en float pour les calculs mathématiques
    l = l.astype(np.float32)

    mean = cv2.GaussianBlur(l, (window_size, window_size), 0)
    l_norm = l - mean
    std = np.sqrt(cv2.GaussianBlur(l_norm ** 2, (window_size, window_size), 0))

    # Normalisation
    l_final = cv2.normalize(l_norm / (std + 1e-5), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Fusion avec les canaux a et b originaux (convertis en uint8 si nécessaire)
    final_img = cv2.cvtColor(cv2.merge((l_final, a, b)), cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return final_img

def read_and_draw_text(image, output_box_path, languages=['en', 'fr'], scale=1.0):

    reader = easyocr.Reader(languages, gpu=False)
    raw_results = reader.readtext(
        image,
        allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -',
        text_threshold=0.7,
    )

    # 2. Preparation
    image_with_boxes = image.copy()

    # IMPORTANT: The scale_factor to return to original coordinates is 1 / scale
    scale_factor = 1.0 / scale
    final_results = []

    for bbox, text, conf in raw_results:
        # Redimensionner les coordonnées pour l'image originale
        bbox_rescaled = [[int(pt[0] * scale_factor), int(pt[1] * scale_factor)] for pt in bbox]
        final_results.append((bbox_rescaled, text, conf))

        # 3. Dessiner
        pts = np.array(bbox_rescaled, dtype=np.int32)
        cv2.polylines(image_with_boxes, [pts], True, (0, 0, 255), 3)
        cv2.putText(image_with_boxes, text, (bbox_rescaled[0][0], bbox_rescaled[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(output_box_path, image_with_boxes, [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
    return final_results

folder_path = r'C:\Users\gab00\OneDrive\Documents\Universite\PMC\Tesseract'
supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm','.pbm')

current_dir = Path(__file__).resolve().parent
assets_dir = current_dir / "assets"
output_base_dir = current_dir / "results"

if output_base_dir.exists():
    print(f"Cleaning up old results: Deleting {output_base_dir}")
    # rmtree deletes the directory and everything inside it
    shutil.rmtree(output_base_dir)

if not assets_dir.exists():
    print(f"Error: assets folder not found at {assets_dir}")
else:
    for filename in os.listdir(assets_dir):
        if filename.lower().endswith(supported_extensions):
            name_root, _ = os.path.splitext(filename)
            image_input_path = assets_dir / filename

            image_result_dir = output_base_dir / name_root
            image_result_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nProcessing: {filename}")
            print(f"Results will be saved in: {image_result_dir}")

            # Define the main output paths inside the new folder
            bbox_output_path = image_result_dir / f"{name_root}_bbox.jpg"
            clean_output_path = image_result_dir / f"{name_root}_cleaned.jpg"
            json_output_path = image_result_dir / f"{name_root}.json"

            # Define the image sharpening comparison paths inside the folder
            output_path = image_result_dir / f"{name_root}.jpg"
            cv2.imwrite(
                str(output_path),
                read_image(str(image_input_path)),
                [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), 0x111111])

            clahe_output_path = image_result_dir / f"{name_root}_clahe.jpg"
            lcn_output_path = image_result_dir / f"{name_root}_lcn.jpg"
            shift_denoising_output_path = image_result_dir / f"{name_root}_shift_denoising.jpg"
            fastnl_denoising_output_path = image_result_dir / f"{name_root}_fastnl_denoising.jpg"
            bilateral_denoising_output_path = image_result_dir / f"{name_root}_bilateral_denoising.jpg"

            clahe_bbox_output_path = image_result_dir / f"{name_root}_clahe_bbox.jpg"
            lcn_bbox_output_path = image_result_dir / f"{name_root}_lcn_bbox.jpg"
            clahe_benchmark_output_path = image_result_dir / f"{name_root}_clahe_benchmark.jpg"
            lcn_benchmark_output_path = image_result_dir / f"{name_root}_lcn_benchmark.jpg"

            #bilateral_denoising_img = bilateral_denoising(str(image_input_path), str(bilateral_denoising_output_path))
            #clean_denoising_img = shift_denoising(str(image_input_path), str(shift_denoising_output_path))
            fastnl_denoising_img = fastnl_denoising(str(image_input_path), str(fastnl_denoising_output_path))
            clahe_img = clahe_sharpening(fastnl_denoising_img.copy(), str(clahe_output_path))
            lcn_img = lcn_sharpening(fastnl_denoising_img.copy(), str(lcn_output_path))

            results_lcn = read_and_draw_text(lcn_img.copy(), str(lcn_bbox_output_path))
            results_clahe = read_and_draw_text(clahe_img.copy(), str(clahe_bbox_output_path))

            #erase_text_lcn(str(image_input_path), results_lcn, str(clean_output_path))
            save_benchmark(str(lcn_benchmark_output_path), results_lcn)
            save_benchmark(str(clahe_benchmark_output_path), results_clahe)

            print(f"Successfully processed {name_root}")
