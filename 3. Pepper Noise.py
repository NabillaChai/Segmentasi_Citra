import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("jalan.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Gambar tidak ditemukan!")
    exit()

gray = img.astype(np.float32)
h, w = gray.shape


# PEPPER NOISE
def manual_pepper_noise(image, prob=0.05):
    noisy = image.copy()

    for y in range(h):
        for x in range(w):
            r = np.random.rand()
            if r < prob:
                noisy[y, x] = 0   # pepper = hitam

    return noisy.astype(np.float32)


pepper_img = manual_pepper_noise(gray, prob=0.05)


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


# METODE ROBERTS
def roberts_operator(img):
    out = np.zeros_like(img)

    for y in range(h - 1):
        for x in range(w - 1):
            z1 = img[y, x]
            z2 = img[y, x+1]
            z3 = img[y+1, x]
            z4 = img[y+1, x+1]

            gx = z2 - z3
            gy = z1 - z4

            out[y, x] = (gx*gx + gy*gy) ** 0.5

    return manual_norm(out)


# METODE PREWITT 
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


# METODE  SOBEL
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
    k = 2 ** 0.5

    for y in range(1, h-1):
        for x in range(1, w-1):

            gx = (img[y-1,x+1] + k*img[y,x+1] + img[y+1,x+1]
                  - img[y-1,x-1] - k*img[y,x-1] - img[y+1,x-1])

            gy = (img[y+1,x-1] + k*img[y+1,x] + img[y+1,x+1]
                  - img[y-1,x-1] - k*img[y-1,x] - img[y-1,x+1])

            out[y, x] = (gx*gx + gy*gy) ** 0.5

    return manual_norm(out)


# PROSES SEMUA METODE
rob_pepper = roberts_operator(pepper_img)
pre_pepper = prewitt_operator(pepper_img)
sob_pepper = sobel_operator(pepper_img)
frei_pepper = frei_chen_operator(pepper_img)


# TAMPILKAN CITRA
plt.figure(figsize=(12, 20))

plt.subplot(3, 2, 1)
plt.title("Original Grayscale")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 2)
plt.title("Pepper Noise")
plt.imshow(pepper_img, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 3)
plt.title("Pepper + Roberts")
plt.imshow(rob_pepper, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 4)
plt.title("Pepper + Prewitt")
plt.imshow(pre_pepper, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 5)
plt.title("Pepper + Sobel")
plt.imshow(sob_pepper, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 6)
plt.title("Pepper + Frei-Chen")
plt.imshow(frei_pepper, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
