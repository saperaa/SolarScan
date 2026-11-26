import numpy as np
import cv2

def negative(img):
    """
    Apply negative transformation to a grayscale image.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    return 255 - img

def log_transform(img):
    """
    Apply logarithmic transformation to a grayscale image.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    img_float = img.astype(np.float64)

    c = 255 / np.log1p(255)
    result = c * np.log1p(img_float)

    return np.clip(result, 0, 255).astype(np.uint8)


def gamma_correction(img, gamma):
    """
    Apply gamma correction to a grayscale image.

    Input: uint8 image (0-255), positive gamma value.
    Output: uint8 image (0-255).
    """
    if gamma <= 0:
        raise ValueError("Gamma must be positive")

    img_float = img.astype(np.float64) / 255.0

    result = 255 * (img_float ** gamma)

    return np.clip(result, 0, 255).astype(np.uint8)


def contrast_stretch(img, low_pct, high_pct):
    """
    Apply contrast stretching using percentile limits.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    low = np.percentile(img, low_pct)
    high = np.percentile(img, high_pct)

    if high <= low:
        return np.zeros_like(img)

    result = (img.astype(np.float64) - low) * 255 / (high - low)

    return np.clip(result, 0, 255).astype(np.uint8)


def histogram_equalization(img):
    """
    Apply histogram equalization using NumPy.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    hist = np.bincount(img.ravel(), minlength=256)

    cdf = hist.cumsum()

    nonzero_cdf = cdf[cdf > 0]

    if len(nonzero_cdf) <= 1:
        return img.copy()

    cdf_min = nonzero_cdf[0]
    total_pixels = img.size

    lookup_table = np.round(
        255 * (cdf - cdf_min) / (total_pixels - cdf_min)
    )

    lookup_table = np.clip(lookup_table, 0, 255).astype(np.uint8)

    return lookup_table[img]


def clahe(img, clip, tiles):
    """
    Apply CLAHE to a grayscale image.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    clahe_obj = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=tiles
    )

    return clahe_obj.apply(img)


def convolve2d(img, kernel, border):
    """
    Apply 2D convolution using NumPy.

    Input: Grayscale image and 2D kernel.
    Output: float64 image.
    Border modes: zero, replicate, reflect.
    """
    img = img.astype(np.float64)
    kernel = np.array(kernel, dtype=np.float64)

    if kernel.ndim != 2 or any(size % 2 == 0 for size in kernel.shape):
        raise ValueError("Kernel dimensions must be odd")

    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    if border == "zero":
        padded = np.pad(
            img, ((pad_h, pad_h), (pad_w, pad_w)),
            mode="constant"
        )
    elif border == "replicate":
        padded = np.pad(
            img, ((pad_h, pad_h), (pad_w, pad_w)),
            mode="edge"
        )
    elif border == "reflect":
        padded = np.pad(
            img, ((pad_h, pad_h), (pad_w, pad_w)),
            mode="reflect"
        )
    else:
        raise ValueError("Invalid border mode")

    flipped_kernel = np.flip(kernel)

    output = np.zeros(img.shape, dtype=np.float64)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[
                i:i + kernel.shape[0],
                j:j + kernel.shape[1]
            ]

            output[i, j] = np.sum(region * flipped_kernel)

    return output


def mean_filter(img, ksize, border="reflect"):
    """
    Apply mean filtering to reduce image noise.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd number")

    kernel = np.ones((ksize, ksize), dtype=np.float64)
    kernel /= (ksize * ksize)

    result = convolve2d(img, kernel, border)

    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def gaussian_filter(img, ksize, sigma, border="reflect"):
    """
    Apply Gaussian filtering to reduce image noise.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd number")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    result = convolve2d(img, kernel, border)

    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def median_smooth(img, ksize, border="reflect"):
    """
    Apply median filtering to remove salt-and-pepper noise.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd number")

    pad = ksize // 2

    padded = np.pad(
        img,
        ((pad, pad), (pad, pad)),
        mode=border if border in ("reflect", "edge") else "constant"
    )

    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + ksize, j:j + ksize]
            output[i, j] = np.median(region)

    return output


def laplacian_sharpen(img, alpha):
    """
    Sharpen a grayscale image using the Laplacian operator.

    Input: uint8 image (0-255), sharpening strength alpha.
    Output: uint8 image (0-255).
    """
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float64)

    laplacian = convolve2d(img, kernel, border="reflect")

    result = img.astype(np.float64) - alpha * laplacian

    return np.clip(result, 0, 255).astype(np.uint8)


def gradient_sharpen(img):
    """
    Sharpen a grayscale image using Sobel gradients.

    Input: uint8 image (0-255).
    Output: uint8 image (0-255).
    """
    gx_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    gy_kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)

    gx = convolve2d(img, gx_kernel, border="reflect")
    gy = convolve2d(img, gy_kernel, border="reflect")

    magnitude = np.sqrt(gx**2 + gy**2)

    result = img.astype(np.float64) + magnitude

    return np.clip(result, 0, 255).astype(np.uint8)
