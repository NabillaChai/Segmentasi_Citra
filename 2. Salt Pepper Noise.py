import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# LOAD & GRAYSCALE
# ==========================================================
img = cv2.imread("jalan.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Gambar tidak ditemukan!")
    exit()

gray = img.astype(np.float32)
h, w = gray.shape


# ==========================================================
# MANUAL SALT & PEPPER NOISE
# ==========================================================
def manual_salt_pepper_noise(image, prob=0.05):
    noisy = image.copy()

    for y in range(h):
        for x in range(w):
            r = np.random.rand()

            if r < prob/2:
                noisy[y, x] = 0       # pepper (hitam)
            elif r < prob:
                noisy[y, x] = 255     # salt (putih)

    return noisy.astype(np.float32)


sp_img = manual_salt_pepper_noise(gray, prob=0.05)


# ==========================================================
# MANUAL NORMALIZATION
# ==========================================================
def manual_norm(img):
    out = np.zeros_like(img)
    maxv = np.max(img)

    if maxv == 0:
        return img.astype(np.uint8)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            out[y, x] = (img[y, x] / maxv) * 255

    return out.astype(np.uint8)


# ==========================================================
# ROBERTS MANUAL
# ==========================================================
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


# ==========================================================
# PREWITT MANUAL
# ==========================================================
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


# ==========================================================
# SOBEL MANUAL
# ==========================================================
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


# ==========================================================
# FREI-CHEN MANUAL
# ==========================================================
def frei_chen_operator(img):
    out = np.zeros_like(img)
    k = np.sqrt(2)

    for y in range(1, h-1):
        for x in range(1, w-1):

            gx = (img[y-1,x+1] + k*img[y,x+1] + img[y+1,x+1]
                  - img[y-1,x-1] - k*img[y,x-1] - img[y+1,x-1])

            gy = (img[y+1,x-1] + k*img[y+1,x] + img[y+1,x+1]
                  - img[y-1,x-1] - k*img[y-1,x] - img[y-1,x+1])

            out[y, x] = (gx*gx + gy*gy) ** 0.5

    return manual_norm(out)


# ==========================================================
# PROSES OPERATOR DENGAN SALT & PEPPER
# ==========================================================
rob_sp = roberts_operator(sp_img)
pre_sp = prewitt_operator(sp_img)
sob_sp = sobel_operator(sp_img)
frei_sp = frei_chen_operator(sp_img)


# ==========================================================
# TAMPILKAN GRID
# ==========================================================
plt.figure(figsize=(12, 20))

plt.subplot(3, 2, 1)
plt.title("Original Grayscale")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 2)
plt.title("Salt & Pepper Noise")
plt.imshow(sp_img, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 3)
plt.title("Salt & Pepper + Roberts")
plt.imshow(rob_sp, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 4)
plt.title("Salt & Pepper + Prewitt")
plt.imshow(pre_sp, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 5)
plt.title("Salt & Pepper + Sobel")
plt.imshow(sob_sp, cmap='gray')
plt.axis("off")

plt.subplot(3, 2, 6)
plt.title("Salt & Pepper + Frei-Chen")
plt.imshow(frei_sp, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
