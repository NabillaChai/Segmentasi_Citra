import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("jalan.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Gambar tidak ditemukan!")
    exit()

gray = img.astype(np.float32)
h, w = gray.shape


# GAUSSIAN NOISE
def manual_gaussian_noise(image, mean=0, sigma=25):
    noisy = np.zeros_like(image)

    for y in range(h):
        for x in range(w):

            u1 = np.random.rand()
            u2 = np.random.rand()

            z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

            noise = z * sigma + mean

            val = image[y, x] + noise

            if val < 0: val = 0
            if val > 255: val = 255

            noisy[y, x] = val

    return noisy.astype(np.float32)


gauss_img = manual_gaussian_noise(gray, sigma=25)


# NORMALISASI CITRA SECARA MANUAL (0–255)
def manual_norm(img):
    out = np.zeros_like(img)
    maxv = np.max(img)

    if maxv == 0:
        return img.astype(np.uint8)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            out[y, x] = (img[y, x] / maxv) * 255

    return out.astype(np.uint8)


# Metode ROBERTS
def roberts_operator(img):
    out = np.zeros_like(img)

    for y in range(h - 1):
        for x in range(w - 1):
            z1 = img[y, x]
            z2 = img[y, x+1]
            z3 = img[y+1, x]
            z4 = img[y+1, x+1]

            gx = z1 - z4
            gy = z3 - z2

            out[y, x] = (gx * gx + gy * gy) ** 0.5

    return manual_norm(out)


# METODE PREWITT MANUAL
def prewitt_operator(img):
    out = np.zeros_like(img)

    for y in range(1, h-1):
        for x in range(1, w-1):

            gx = (img[y-1,x-1] + img[y,x-1] + img[y+1,x-1]
                  - img[y-1,x+1] - img[y,x+1] - img[y+1,x+1])

            gy = (img[y+1,x-1] + img[y+1,x] + img[y+1,x+1]
                  - img[y-1,x-1] - img[y-1,x] - img[y-1,x+1])

            out[y, x] = (gx*gx + gy*gy) ** 0.5

    return manual_norm(out)


# METODE SOBEL 
def sobel_operator(img):
    out = np.zeros_like(img)

    for y in range(1, h-1):
        for x in range(1, w-1):

            gx = (img[y-1,x+1] + 2*img[y,x+1] + img[y+1,x+1]
                  - img[y-1,x-1] - 2*img[y,x-1] - img[y+1,x-1])

            gy = (img[y+1,x-1] + 2*img[y+1,x] + img[y+1,x+1]
                  - img[y-1,x-1] - 2*img[y-1,x] - img[y-1,x+1])

            out[y, x] = (gx*gx + gy*gy) ** 0.5

    return manual_norm(out)


# METODE FREI-CHEN
def frei_chen_operator(img):
    out = np.zeros_like(img)
    k = 2 ** 0.5  # sqrt(2)

    for y in range(1, h-1):
        for x in range(1, w-1):

            gx = (img[y-1,x+1] + k*img[y,x+1] + img[y+1,x+1]
                  - img[y-1,x-1] - k*img[y,x-1] - img[y+1,x-1])

            gy = (img[y+1,x-1] + k*img[y+1,x] + img[y+1,x+1]
                  - img[y-1,x-1] - k*img[y-1,x] - img[y-1,x+1])

            out[y, x] = (gx*gx + gy*gy) ** 0.5

    return manual_norm(out)

# PROSES SEMUA METODE
rob_gauss = roberts_operator(gauss_img)
pre_gauss = prewitt_operator(gauss_img)
sob_gauss = sobel_operator(gauss_img)
frei_gauss = frei_chen_operator(gauss_img)


# TAMPILKAN CITRA
plt.figure(figsize=(12, 20))

plt.subplot(3, 2, 1)
plt.title("Original Grayscale")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 2)
plt.title("Gaussian Noise")
plt.imshow(gauss_img, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 3)
plt.title("Gaussian + Roberts")
plt.imshow(rob_gauss, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 4)
plt.title("Gaussian + Prewitt")
plt.imshow(pre_gauss, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 5)
plt.title("Gaussian + Sobel")
plt.imshow(sob_gauss, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 6)
plt.title("Gaussian + Frei-Chen")
plt.imshow(frei_gauss, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
