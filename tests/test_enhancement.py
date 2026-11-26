import numpy as np
import cv2
import pytest

from src.enhancement import (
    negative,
    log_transform,
    gamma_correction,
    contrast_stretch,
    histogram_equalization,
    clahe,
    convolve2d,
    mean_filter,
    gaussian_filter,
    median_smooth,
    laplacian_sharpen,
    gradient_sharpen,
    flatten_background,
)


# --------------------------------------------------
# 1. Negative
# --------------------------------------------------

def test_negative():
    img = np.array([[0, 50, 100, 255]], dtype=np.uint8)

    result = negative(img)

    expected = np.array([[255, 205, 155, 0]], dtype=np.uint8)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint8


# --------------------------------------------------
# 2. Log Transformation
# --------------------------------------------------

def test_log_transform():
    img = np.array([[0, 50, 100, 200, 255]], dtype=np.uint8)

    result = log_transform(img)

    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255

    # Log transformation should preserve intensity order
    assert np.all(np.diff(result[0]) >= 0)
    assert result[0, 0] == 0
    assert result[0, -1] == 255


# --------------------------------------------------
# 3. Gamma Correction
# --------------------------------------------------

def test_gamma_correction():
    img = np.array([[0, 64, 128, 192, 255]], dtype=np.uint8)

    result = gamma_correction(img, 1.0)

    np.testing.assert_array_equal(result, img)
    assert result.dtype == np.uint8


def test_gamma_correction_invalid_gamma():
    img = np.array([[0, 100, 255]], dtype=np.uint8)

    with pytest.raises(ValueError):
        gamma_correction(img, 0)


# --------------------------------------------------
# 4. Contrast Stretching
# --------------------------------------------------

def test_contrast_stretch():
    img = np.array([[50, 100, 150, 200]], dtype=np.uint8)

    result = contrast_stretch(img, 0, 100)

    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_contrast_stretch_constant_image():
    img = np.full((4, 4), 100, dtype=np.uint8)

    result = contrast_stretch(img, 0, 100)

    assert np.all(result == 0)


# --------------------------------------------------
# 5. Histogram Equalization
# --------------------------------------------------

def test_histogram_equalization():
    img = np.array(
        [[0, 0, 1, 1, 2, 2, 3, 3]],
        dtype=np.uint8
    )

    result = histogram_equalization(img)

    reference = cv2.equalizeHist(img)

    assert result.shape == img.shape
    assert result.dtype == np.uint8

    # NumPy implementation should be very close to OpenCV
    assert np.max(np.abs(
        result.astype(np.int16) -
        reference.astype(np.int16)
    )) <= 1


def test_histogram_equalization_constant_image():
    img = np.full((4, 4), 120, dtype=np.uint8)

    result = histogram_equalization(img)

    np.testing.assert_array_equal(result, img)


# --------------------------------------------------
# 6. CLAHE
# --------------------------------------------------

def test_clahe():
    img = np.array(
        [[10, 20, 30, 40],
         [50, 60, 70, 80],
         [90, 100, 110, 120],
         [130, 140, 150, 160]],
        dtype=np.uint8
    )

    result = clahe(img, 2.0, (2, 2))

    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


# --------------------------------------------------
# 7. 2D Convolution
# --------------------------------------------------

def test_convolve2d_identity_kernel():
    img = np.array(
        [[10, 20, 30],
         [40, 50, 60],
         [70, 80, 90]],
        dtype=np.uint8
    )

    kernel = np.array(
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]
    )

    result = convolve2d(img, kernel, "zero")

    np.testing.assert_array_equal(result, img.astype(np.float64))
    assert result.dtype == np.float64


def test_convolve2d_kernel_is_flipped():
    img = np.array(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        dtype=np.uint8
    )

    kernel = np.array(
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    )

    result = convolve2d(img, kernel, "zero")

    # Convolution flips the kernel
    assert result[1, 1] == 9


@pytest.mark.parametrize("border", ["zero", "replicate", "reflect"])
def test_convolve2d_border_modes(border):
    img = np.array(
        [[10, 20, 30],
         [40, 50, 60],
         [70, 80, 90]],
        dtype=np.uint8
    )

    kernel = np.array(
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]
    )

    result = convolve2d(img, kernel, border)

    assert result.shape == img.shape
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, img.astype(np.float64))

@pytest.mark.parametrize(
    "border, cv_border",
    [
        ("zero", cv2.BORDER_CONSTANT),
        ("replicate", cv2.BORDER_REPLICATE),
        ("reflect", cv2.BORDER_REFLECT_101),
    ]
)
def test_convolve2d_matches_opencv(border, cv_border):
    img = np.array(
        [[10, 20, 30, 40],
         [50, 60, 70, 80],
         [90, 100, 110, 120],
         [130, 140, 150, 160]],
        dtype=np.uint8
    )

    kernel = np.array(
        [[1, 2, 0],
         [0, 1, 0],
         [0, 0, -1]],
        dtype=np.float64
    )

    result = convolve2d(img, kernel, border)

    # OpenCV filter2D performs correlation, so flip the kernel
    # to compare it with our convolution implementation.
    expected = cv2.filter2D(
        img.astype(np.float64),
        cv2.CV_64F,
        np.flip(kernel),
        borderType=cv_border
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-10,
        atol=1e-10
    )
# --------------------------------------------------
# 8. Mean Filter
# --------------------------------------------------

def test_mean_filter():
    img = np.full((5, 5), 100, dtype=np.uint8)

    result = mean_filter(img, 3)

    assert result.shape == img.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, img)


# --------------------------------------------------
# 9. Gaussian Filter
# --------------------------------------------------

def test_gaussian_filter():
    img = np.full((5, 5), 100, dtype=np.uint8)

    result = gaussian_filter(img, 3, 1.0)

    assert result.shape == img.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, img)


# --------------------------------------------------
# 10. Median Smoothing
# --------------------------------------------------

def test_median_smooth():
    img = np.full((5, 5), 100, dtype=np.uint8)

    # Add a noisy pixel
    img[2, 2] = 255

    result = median_smooth(img, 3)

    assert result.shape == img.shape
    assert result.dtype == np.uint8

    # Median filter should remove the isolated noisy pixel
    assert result[2, 2] == 100


# --------------------------------------------------
# 11. Laplacian Sharpening
# --------------------------------------------------

def test_laplacian_sharpen():
    img = np.full((5, 5), 100, dtype=np.uint8)

    result = laplacian_sharpen(img, 1.0)

    assert result.shape == img.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, img)


# --------------------------------------------------
# 12. Gradient Sharpening
# --------------------------------------------------

def test_gradient_sharpen():
    img = np.full((5, 5), 100, dtype=np.uint8)

    result = gradient_sharpen(img)

    assert result.shape == img.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, img)


# --------------------------------------------------
# 13. Background Flattening
# --------------------------------------------------

def test_flatten_background():
    img = np.full((5, 5), 100, dtype=np.uint8)

    result = flatten_background(img, 3)

    assert result.shape == img.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, img)


# --------------------------------------------------
# General Output Validation
# --------------------------------------------------

def test_all_functions_output_range():
    img = np.array(
        [[0, 50, 100],
         [150, 200, 255],
         [30, 80, 130]],
        dtype=np.uint8
    )

    results = [
        negative(img),
        log_transform(img),
        gamma_correction(img, 1.5),
        contrast_stretch(img, 0, 100),
        histogram_equalization(img),
        clahe(img, 2.0, (2, 2)),
        mean_filter(img, 3),
        gaussian_filter(img, 3, 1.0),
        median_smooth(img, 3),
        laplacian_sharpen(img, 1.0),
        gradient_sharpen(img),
        flatten_background(img, 3),
    ]

    for result in results:
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255


@pytest.mark.parametrize(
    ("border", "cv_border"),
    [
        ("zero", cv2.BORDER_CONSTANT),
        ("replicate", cv2.BORDER_REPLICATE),
        ("reflect", cv2.BORDER_REFLECT_101),
    ],
)
def test_convolve2d_matches_opencv(border, cv_border):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (20, 24), dtype=np.uint8)
    kernel = rng.random((5, 5))

    result = convolve2d(img, kernel, border)
    # cv2.filter2D computes correlation, so flip the kernel to get convolution.
    expected = cv2.filter2D(
        img.astype(np.float64), -1, np.flip(kernel), borderType=cv_border
    )

    assert np.abs(result - expected).max() < 1e-9


def test_convolve2d_mean_kernel_hand_worked():
    # Worked on paper, 3x3 mean kernel (every weight 1/9), zero border:
    #   centre (1, 1): (10+20+30+40+50+60+70+80+90) / 9 = 450 / 9 = 50
    #   corner (0, 0): only 10, 20, 40, 50 are inside -> 120 / 9 = 13.33
    img = np.array(
        [[10, 20, 30],
         [40, 50, 60],
         [70, 80, 90]],
        dtype=np.uint8
    )
    kernel = np.full((3, 3), 1 / 9)

    result = convolve2d(img, kernel, "zero")

    assert result[1, 1] == pytest.approx(50.0)
    assert result[0, 0] == pytest.approx(120 / 9)
