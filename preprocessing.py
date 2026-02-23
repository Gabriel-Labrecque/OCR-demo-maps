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

    model_name = f"EDSR_x{scale}.pb"
    if not os.path.exists(model_name):
        print(f"Downloading {model_name}...")
        urllib.request.urlretrieve(
            f"https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/{model_name}",
            model_name
        )

    # EDSR expects uint8 BGR
    img_uint8 = skimage.util.img_as_ubyte(img)
    img_bgr = img_uint8[:, :, ::-1]

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_name)
    sr.setModel("edsr", scale)
    upscaled_bgr = sr.upsample(img_bgr)

    # convert back to float64 RGB [0.0, 1.0]
    upscaled_rgb = upscaled_bgr[:, :, ::-1]
    return skimage.util.img_as_float(upscaled_rgb)

def clahe_color_amplification(img: np.ndarray, amplification: float = 0.03) -> np.ndarray:

    img = np.clip(img.astype(np.float64), 0.0, 1.0)
    img_lab = skimage.color.rgb2lab(img)
    l_channel = np.clip(img_lab[:, :, 0] / 100.0, 0.0, 1.0)

    l_enhanced = skimage.exposure.equalize_adapthist(
        l_channel,
        kernel_size=None,
        clip_limit=amplification
    )

    img_lab[:, :, 0] = l_enhanced * 100.0
    img_lab[:, :, 0] = np.clip(img_lab[:, :, 0], 0.0, 100.0)
    img_lab[:, :, 1] = np.clip(img_lab[:, :, 1], -128.0, 127.0)
    img_lab[:, :, 2] = np.clip(img_lab[:, :, 2], -128.0, 127.0)

    img_rgb = skimage.color.lab2rgb(img_lab)
    return np.clip(img_rgb, 0.0, 1.0)


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


def denoise_image(img: np.ndarray, amplitude: float = 0.1) -> np.ndarray:
    # Estimate noise standard deviation from the image
    sigma_est = np.mean(skimage.restoration.estimate_sigma(img, channel_axis=-1))
    # Apply Non-Local Means denoising
    return skimage.restoration.denoise_nl_means(img, h=amplitude * sigma_est,
                                                sigma=sigma_est,
                                                fast_mode=True,
                                                channel_axis=-1
                                                )

def bilateral_denoise(img: np.ndarray, sigma_color: float = 0.05, sigma_spatial: float = 1.0) -> np.ndarray:
    return skimage.restoration.denoise_bilateral(img,
                                                  sigma_color=sigma_color,
                                                  sigma_spatial=sigma_spatial,
                                                  channel_axis=-1)


def prepare_for_ocr(img: np.ndarray) -> np.ndarray:
    # Swaps from RGB to BGR and cast pixel values are int
    if img.dtype != np.uint8:
        img = skimage.util.img_as_ubyte(img)

    return img[:, :, ::-1]
