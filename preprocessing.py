import numpy as np
import skimage
import os

def read_image(image_path) -> np.ndarray:
    """
    Read image and formats to RGB float64 format
    :param image_path: Path to the image
    :return: 3 channels RGB formatted numpy array of the image. Values are floats [0.0, 1.0]
    :raises: IOError
    """
    # OpenCV reads in RGB format
    img = skimage.io.imread(image_path)
    if img is None:
        raise IOError(f'Could not read image for given path: {image_path}')

    # Reads the saved image and normalize to float64
    img = skimage.util.img_as_float(img)

    # Greyscale if the array is 2D
    if img.ndim == 2:
        img = skimage.color.gray2rgb(img)

    # Multichannel if the array is 3D
    elif img.ndim == 3:
        channels = img.shape[2]

        # Assume a 4th channel is opacity
        if channels == 4:

            alpha = img[:, :, 3:4]
            rgb = img[:, :, :3]

            white_bg = np.ones_like(rgb)
            img = (rgb * alpha) + (white_bg * (1 - alpha))

        # More than 4 channels is useless and removed
        elif channels > 4:
            img = img[:, :, :3]

    return img


def scale_image(img: np.ndarray, width: int = 0, height: int = 0) -> np.ndarray:
    """
    Rescales an RGB image to a given width and height. Untested for LAB image formats
    param img: RGB image with each os value in a range of [0.0, 1.0]
    param width: width of the new image
    param height: height of the new image
    return: scaled image: image composed of 3 RGB channels, float values between [0.0, 1.0]
    """

    if width < 0 or height < 0:
        raise ValueError(f"Target dimensions must be >= 0. Width: {width}, height: {height}.")
    if width == 0 or height == 0:
        raise ValueError(f"Width and height must be specified for scaling image. Width: {width}, height: {height}.")

    output_shape = (height, width)

    rescaled = skimage.transform.resize(
        img, output_shape,
        order=4,            # Lanczos interpolation for new lines
        mode='reflect',     # Mimics the surrounding color gradients
        anti_aliasing=True, # Only used IF the wanted dimensions are smaller than the actual image
        preserve_range=True # Keeping as float64
    )

    # Prevent negative values and values over 1 that comes from using float64
    return np.clip(rescaled, 0.0, 1.0, out=rescaled)


def upscale_lanczos(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Upscales the image using Lanczos-4 interpolation to preserve text sharpness.
    :param img: RGB image with float values between [0.0, 1.0]
    :param scale: Upscale factor (e.g. 2 = double the size)
    :return: Upscaled image with float values between [0.0, 1.0]
    """
    new_height = int(img.shape[0] * scale)
    new_width = int(img.shape[1] * scale)
    return scale_image(img, width=new_width, height=new_height)


def upscale_ai(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Upscales the image using EDSR super-resolution model.
    :param img: RGB image with float values between [0.0, 1.0]
    :param scale: Upscale factor — must be 2, 3, or 4
    :return: Upscaled image with float values between [0.0, 1.0]
    """
    import cv2
    import urllib.request
    from cv2 import dnn_superres

    if scale not in (2, 3, 4):
        raise ValueError(f"AI upscaling only supports scale 2, 3, or 4. Got: {scale}")

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"EDSR_x{scale}.pb")

    if not os.path.exists(model_path):
        print(f"Downloading EDSR_x{scale}.pb...")
        urllib.request.urlretrieve(
            f"https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x{scale}.pb",
            model_path
        )

    # EDSR expects uint8 BGR
    img_uint8 = skimage.util.img_as_ubyte(np.clip(img, 0.0, 1.0))
    img_bgr = img_uint8[:, :, ::-1]

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", scale)
    upscaled_bgr = sr.upsample(img_bgr)

    # convert back to float64 RGB [0.0, 1.0]
    upscaled_rgb = upscaled_bgr[:, :, ::-1]
    return skimage.util.img_as_float(upscaled_rgb)

def clahe_color_amplification(img: np.ndarray, amplification: float = 0.3) -> np.ndarray:
    img = np.clip(img.astype(np.float64), 0.0, 1.0)

    img_hsv = skimage.color.rgb2hsv(img)
    v_channel = img_hsv[:, :, 2]  # Already [0.0, 1.0]

    v_enhanced = skimage.exposure.equalize_adapthist(
        v_channel,
        kernel_size=None,
        clip_limit=0.03
    )

    # Blend enhanced V with original to control intensity
    img_hsv[:, :, 2] = np.clip(
        (1.0 - amplification) * v_channel + amplification * v_enhanced,
        0.0, 1.0
    )

    return np.clip(skimage.color.hsv2rgb(img_hsv), 0.0, 1.0)


def gamma_correction(img: np.ndarray, gamma:float=0.7, amplitude:float=0.5):
    """
    Apply a `gamma` exponent to the L channel of every pixel. The amplitude is
    a linear parameter that simply blends the result with the existing values

    :param img:RGB image with float values between [0.0, 1.0]
    :param gamma: The power-law exponent. 0.5 < `gamma` < 1 makes it lighter, 1 < `gamma` < 2 makes it darker.
    Keeping this parameter close to 1 prevent it from being too aggressive
    :param amplitude: The % intensity of the effect (0.0 = no effect, 1.0 = pure gamma result)
    :return: Image with corrected gamma. Values in range [0.0, 1.0]
    """
    if gamma < 0.5 or gamma > 2:
        raise ValueError(f"gamma must be between 0.5 and 2.0. Gamma: {gamma}")

    img_lab = skimage.color.rgb2lab(img)  # L=[0.0, 100.0]
    l_channel = img_lab[:, :, 0] / 100.0  # L=[0.0, 1.0]

    l_gamma = np.power(l_channel, gamma)
    l_gamma = (1.0 - amplitude) * l_channel + amplitude * l_gamma
    img_lab[:, :, 0] = l_gamma * 100.0

    img = skimage.color.lab2rgb(img_lab)
    return np.clip(img, 0.0, 1.0)


def lcn_sharpening_skimage(img: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Local Contrast Normalization (LCN) using scikit-image. Acts more or less
    like a high-pass filter

    :param img: Input RGB image (float64, range [0.0, 1.0])
    :param window_size: Size of the Gaussian kernel (must be odd)
    :return: LCN enhanced RGB image with values in the [0.1, 1.0] float64 range
    """
    # Convert to LAB and extract L
    img_float = skimage.util.img_as_float(img)
    lab = skimage.color.rgb2lab(img_float)
    l_channel = lab[:, :, 0]  # Range [0, 100]

    # 2. Local Mean Estimation (average adjacent pixel brightness)
    # sigma is arbitrarily `window_size / 6` for equivalent OpenCV behavior
    sigma = window_size / 6.0
    local_mean = skimage.filters.gaussian(l_channel, sigma=sigma, mode='reflect')
    l_subtracted = l_channel - local_mean

    # E[X^2] - (E[X])^2 approach is faster, but your code uses E[(X-mu)^2]
    local_var = skimage.filters.gaussian(l_subtracted ** 2, sigma=sigma, mode='reflect')
    local_std = np.sqrt(local_var)

    # 4. Normalization (The Scaling Stage)
    # We add a epsilon (1e-5) to prevent division by zero in flat areas
    l_normalized = l_subtracted / (local_std + 1e-5)

    # 5. Range Rescaling
    # LCN results in a zero-centered signal. For LAB, we must map these.
    l_final = skimage.exposure.rescale_intensity(l_normalized, out_range=(0, 100))

    # 6. Reconstruct
    lab[:, :, 0] = l_final
    result_rgb = skimage.color.lab2rgb(lab)

    return np.clip(result_rgb, 0.0, 1.0)


def denoise_meanshift(img: np.ndarray, spatial_radius: int = 10, color_radius: float = 0.10) -> np.ndarray:
    """
    Mean shift filter — smooths background texture while preserving text edges.
    :param img: RGB float64 [0.0, 1.0]
    :param spatial_radius: How far spatially to look for similar pixels (in pixels)
    :param color_radius: How similar in color pixels need to be to be grouped [0.0, 1.0]
    :return: Filtered RGB float64 [0.0, 1.0]
    """
    import cv2

    img_uint8 = skimage.util.img_as_ubyte(img)
    img_bgr = img_uint8[:, :, ::-1]  # RGB -> BGR for OpenCV

    # color_radius is [0,1] but OpenCV expects [0,255]
    filtered_bgr = cv2.pyrMeanShiftFiltering(
        img_bgr,
        sp=spatial_radius,
        sr=int(color_radius * 255)
    )

    filtered_rgb = filtered_bgr[:, :, ::-1]  # BGR -> RGB
    return skimage.util.img_as_float(filtered_rgb)


def denoise_image(img: np.ndarray, amplitude: float = 0.1) -> np.ndarray:
    # Estimate noise standard deviation from the image
    sigma_est = np.mean(skimage.restoration.estimate_sigma(img, channel_axis=-1))
    # Apply Non-Local Means denoising
    return skimage.restoration.denoise_nl_means(img, h=amplitude * sigma_est,
                                                sigma=sigma_est,
                                                fast_mode=True,
                                                channel_axis=-1
                                                )


def denoise_ai(img: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    import torch
    import torch.nn as nn
    import urllib.request

    class DnCNN(nn.Module):
        def __init__(self, channels=3, num_layers=20):
            super().__init__()
            layers = []
            for _ in range(num_layers - 1):
                layers += [nn.Conv2d(channels if _ == 0 else 64, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(64, channels, 3, padding=1, bias=True)]
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return x - self.model(x)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "dncnn_color_blind.pth")

    if not os.path.exists(model_path):
        print("Downloading DnCNN color blind weights...")
        urllib.request.urlretrieve(
            "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth",
            model_path
        )

    state_dict = torch.load(model_path, map_location="cpu")

    # KAIR state dict keys are prefixed with "module." when saved with DataParallel — strip if needed
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = DnCNN(channels=3)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    img_f32 = img.astype(np.float32)
    tensor = torch.from_numpy(img_f32.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        denoised = model(tensor).squeeze(0).numpy().transpose(1, 2, 0)

    blended = (1.0 - amplitude) * img + amplitude * denoised.astype(np.float64)
    return np.clip(blended, 0.0, 1.0)


def bilateral_denoise(img: np.ndarray, sigma_color: float = 0.05, sigma_spatial: float = 1.0) -> np.ndarray:
    return skimage.restoration.denoise_bilateral(img,
                                                  sigma_color=sigma_color,
                                                  sigma_spatial=sigma_spatial,
                                                  channel_axis=-1)


def color_equalization(img: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """
    Equalizes luminance only (L channel in LAB space).
    Colors are completely preserved, only brightness contrast is enhanced.
    :param img: RGB float64 [0.0, 1.0]
    :param clip_limit: CLAHE aggressiveness (0.01 subtle, 0.05 strong)
    :return: RGB float64 [0.0, 1.0]
    """
    img = np.clip(img.astype(np.float64), 0.0, 1.0)
    img_lab = skimage.color.rgb2lab(img)

    l_channel = np.clip(img_lab[:, :, 0] / 100.0, 0.0, 1.0)
    l_enhanced = skimage.exposure.equalize_adapthist(l_channel, clip_limit=clip_limit)

    img_lab[:, :, 0] = np.clip(l_enhanced * 100.0, 0.0, 100.0)
    return np.clip(skimage.color.lab2rgb(img_lab), 0.0, 1.0)


def color_quantization(img: np.ndarray, n_colors: int = 12) -> np.ndarray:
    """
    Reduces the image to n_colors using K-means clustering.
    Flattens background gradients into uniform regions so text stands out.
    :param img: RGB float64 [0.0, 1.0]
    :param n_colors: Number of distinct colors to keep (8-16 works well for maps)
    :return: RGB float64 [0.0, 1.0]
    """
    import cv2

    img_uint8 = skimage.util.img_as_ubyte(img)
    pixels = img_uint8.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    quantized = np.uint8(centers)[labels.flatten()].reshape(img_uint8.shape)
    return skimage.util.img_as_float(quantized)


def prepare_for_ocr(img: np.ndarray) -> np.ndarray:
    # Swaps from RGB to BGR and cast pixel values are int
    if img.dtype != np.uint8:
        img = skimage.util.img_as_ubyte(img)

    return img[:, :, ::-1]
