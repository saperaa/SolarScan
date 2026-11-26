# 3×3 Mean Filter – Hand Calculations

A 3×3 Mean Filter replaces each pixel with the average of the nine pixel values in its 3×3 neighborhood.

## Input

```
image = [[10, 20, 30],          kernel = 1/9 · [[1, 1, 1],
         [40, 50, 60],                          [1, 1, 1],
         [70, 80, 90]]                          [1, 1, 1]]
```

Border handling: **zero padding** (pixels outside the image count as 0). These values are
checked by `test_convolve2d_mean_kernel_hand_worked` in `tests/test_enhancement.py`.

## 1. Centre Pixel Calculation

The 3×3 window around (1, 1) is the whole image, all nine values real:

Sum = 10 + 20 + 30 + 40 + 50 + 60 + 70 + 80 + 90 = 450.

Mean = 450 / 9 = 50.

**Result: Centre Pixel = 50**

## 2. Corner Pixel Calculation

The 3×3 window around (0, 0) holds only **4 real pixels** (10, 20, 40, 50); the other 5
positions fall outside the image and are padded with 0:

```
 0   0   0
 0  10  20
 0  40  50
```

Sum = 10 + 20 + 40 + 50 = 120. The divisor stays 9 (the kernel weight is 1/9 everywhere).

Mean = 120 / 9 ≈ 13.33.

**Result: Corner Pixel ≈ 13.33**

With **replicate** padding the window would be `10 10 20 / 10 10 20 / 40 40 50`
(sum 210, mean 23.33), so the border mode changes the result at the edges.

## 3. Conclusion

The 3×3 Mean Filter smooths an image by replacing each pixel with the average intensity of its neighborhood. This reduces local intensity variations and helps suppress noise. Near the borders the result depends on the padding mode, which is why every neighbourhood operation in SolarScan states its border handling.
