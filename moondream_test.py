import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from skimage import io
from PIL import Image

os.environ["HF_HOME"] = "./models"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map={"": "cpu"}  # no GPU
)
tokenizer = AutoTokenizer.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21"
)
print("Model ready.")

for filename in os.listdir("input/"):
    if os.path.splitext(filename)[1].lower() not in SUPPORTED_EXTENSIONS:
        continue

    print(f"Processing {filename}...")
    img_array = io.imread(os.path.join("input/", filename))
    if img_array.ndim == 3 and img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    image = Image.fromarray(img_array)  # moondream still needs PIL internally
    result = model.query(image, "List all text labels visible on this map.")["answer"]

    print(f"--- {filename} ---")
    print(result)