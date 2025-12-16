import cv2
import numpy as np
import matplotlib.pyplot as plt

# SET SEED AGAR NILAI MSE TETAP SAMA
np.random.seed(42)

# LOAD GAMBAR ASLI (BERWARNA)
img_asli = cv2.imread("patung2.jpg")
if img_asli is None:
    print("Gambar tidak ditemukan!")
    exit()

# Konversi BGR ke RGB 
img_asli_rgb = cv2.cvtColor(img_asli, cv2.COLOR_BGR2RGB)

# KONVERSI KE GRAYSCALE SECARA MANUAL
print("Mengkonversi gambar ke grayscale...")
h, w = img_asli.shape[:2]
gray = np.zeros((h, w), dtype=np.float32)

# Konversi manual RGB ke Grayscale: 0.299*R + 0.587*G + 0.114*B
for y in range(h):
    for x in range(w):
        b, g, r = img_asli[y, x]
        gray[y, x] = 0.299 * r + 0.587 * g + 0.114 * b


# FUNGSI HITUNG MSE SECARA MANUAL
def hitung_mse(citra_asli, citra_noise):  
    # Konversi ke float untuk perhitungan
    citra_asli = citra_asli.astype(np.float32)
    citra_noise = citra_noise.astype(np.float32)
    
    # Mendapatkan dimensi matriks
    M = citra_asli.shape[0]  # jumlah baris
    N = citra_asli.shape[1]  # jumlah kolom
    
    # Inisialisasi variabel untuk menyimpan jumlah kuadrat selisih
    jumlah_kuadrat_selisih = 0.0
    
    # Loop untuk setiap piksel (i,j) 
    for i in range(M):
        for j in range(N):
            # Hitung selisih antara citra asli dan citra dengan noise
            selisih = citra_asli[i][j] - citra_noise[i][j]
            
            # Kuadratkan selisih secara manual
            kuadrat_selisih = selisih * selisih
            
            # Tambahkan ke jumlah total
            jumlah_kuadrat_selisih += kuadrat_selisih
    
    # Hitung MSE dengan membagi jumlah kuadrat selisih dengan M*N
    mse = jumlah_kuadrat_selisih / (M * N)
    
    return mse

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


# MANUAL GAUSSIAN NOISE
def manual_gaussian_noise(image, mean=0, sigma=25):
    noisy = np.zeros_like(image)
    
    for y in range(h):
        for x in range(w):
            # Box-Muller Transform untuk generate Gaussian distribution
            u1 = np.random.rand()
            u2 = np.random.rand()
            
            # Hitung manual sqrt, log, cos
            z = ((-2 * np.log(u1)) ** 0.5) * np.cos(2 * 3.14159265359 * u2)
            
            # Tambahkan noise
            noise = z * sigma + mean
            val = image[y, x] + noise
            
            # Clipping manual
            if val < 0: 
                val = 0
            if val > 255: 
                val = 255
            
            noisy[y, x] = val
    
    return noisy.astype(np.float32)


# MANUAL SALT & PEPPER NOISE
def manual_salt_pepper_noise(image, prob=0.05):
    """
    Membuat Salt & Pepper Noise secara manual
    """
    noisy = image.copy()
    
    for y in range(h):
        for x in range(w):
            r = np.random.rand()
            
            # Pepper (hitam)
            if r < prob/2:
                noisy[y, x] = 0
            # Salt (putih)
            elif r < prob:
                noisy[y, x] = 255
    
    return noisy.astype(np.float32)


# FILTER MEDIAN MANUAL 5x5
def median_filter_manual(img):
    out = np.zeros_like(img)
    padded = np.zeros((h+4, w+4), dtype=np.float32)
    padded[2:-2, 2:-2] = img
    
    for y in range(h):
        for x in range(w):
            kernel = padded[y:y+5, x:x+5].flatten()
            kernel_sorted = np.sort(kernel)
            out[y, x] = kernel_sorted[len(kernel_sorted)//2]
    
    return out


# FILTER MEAN MANUAL 5x5
def mean_filter_manual(img):
    out = np.zeros_like(img)
    padded = np.zeros((h+4, w+4), dtype=np.float32)
    padded[2:-2, 2:-2] = img
    
    for y in range(h):
        for x in range(w):
            kernel = padded[y:y+5, x:x+5]
            out[y, x] = np.sum(kernel) / 25
    
    return out


# METODE ROBERTS
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
    k = np.sqrt(2)
    for y in range(1, h-1):
        for x in range(1, w-1):
            gx = (img[y-1,x+1] + k*img[y,x+1] + img[y+1,x+1]
                  - img[y-1,x-1] - k*img[y,x-1] - img[y+1,x-1])
            gy = (img[y+1,x-1] + k*img[y+1,x] + img[y+1,x+1]
                  - img[y-1,x-1] - k*img[y-1,x] - img[y-1,x+1])
            out[y, x] = (gx*gx + gy*gy) ** 0.5
    return manual_norm(out)


# GENERATE NOISE, FILTER, DAN SEGMENTASI
print("Memproses Noise, Filter, dan Segmentasi...")

# Generate noise
sp_img = manual_salt_pepper_noise(gray, prob=0.05)
gauss_img = manual_gaussian_noise(gray, sigma=25)

# Apply filter pada Salt & Pepper
sp_median = median_filter_manual(sp_img)
sp_mean = mean_filter_manual(sp_img)

# Segmentasi untuk grayscale asli
print("  - Segmentasi Grayscale Asli...")
gray_roberts = roberts_operator(gray)
gray_prewitt = prewitt_operator(gray)
gray_sobel = sobel_operator(gray)
gray_frei = frei_chen_operator(gray)

# Segmentasi untuk Salt & Pepper Noise
print("  - Segmentasi Salt & Pepper Noise...")
sp_roberts = roberts_operator(sp_img)
sp_prewitt = prewitt_operator(sp_img)
sp_sobel = sobel_operator(sp_img)
sp_frei = frei_chen_operator(sp_img)

# Segmentasi untuk Gaussian Noise
print("  - Segmentasi Gaussian Noise...")
gauss_roberts = roberts_operator(gauss_img)
gauss_prewitt = prewitt_operator(gauss_img)
gauss_sobel = sobel_operator(gauss_img)
gauss_frei = frei_chen_operator(gauss_img)

# Segmentasi untuk Salt & Pepper + Median Filter
print("  - Segmentasi Salt & Pepper + Median Filter...")
sp_median_roberts = roberts_operator(sp_median)
sp_median_prewitt = prewitt_operator(sp_median)
sp_median_sobel = sobel_operator(sp_median)
sp_median_frei = frei_chen_operator(sp_median)

# Segmentasi untuk Salt & Pepper + Mean Filter
print("  - Segmentasi Salt & Pepper + Mean Filter...")
sp_mean_roberts = roberts_operator(sp_mean)
sp_mean_prewitt = prewitt_operator(sp_mean)
sp_mean_sobel = sobel_operator(sp_mean)
sp_mean_frei = frei_chen_operator(sp_mean)


# HITUNG MSE UNTUK SEMUA KOMBINASI
print("\nMenghitung MSE untuk semua kombinasi...")

# 1. Grayscale tersegmentasi vs Salt & Pepper Noise tersegmentasi
mse_sp_roberts = hitung_mse(gray_roberts.astype(np.float32), sp_roberts.astype(np.float32))
mse_sp_prewitt = hitung_mse(gray_prewitt.astype(np.float32), sp_prewitt.astype(np.float32))
mse_sp_sobel = hitung_mse(gray_sobel.astype(np.float32), sp_sobel.astype(np.float32))
mse_sp_frei = hitung_mse(gray_frei.astype(np.float32), sp_frei.astype(np.float32))

# 2. Grayscale tersegmentasi vs Gaussian Noise tersegmentasi
mse_gauss_roberts = hitung_mse(gray_roberts.astype(np.float32), gauss_roberts.astype(np.float32))
mse_gauss_prewitt = hitung_mse(gray_prewitt.astype(np.float32), gauss_prewitt.astype(np.float32))
mse_gauss_sobel = hitung_mse(gray_sobel.astype(np.float32), gauss_sobel.astype(np.float32))
mse_gauss_frei = hitung_mse(gray_frei.astype(np.float32), gauss_frei.astype(np.float32))

# 3. Grayscale tersegmentasi vs Salt & Pepper + Median Filter tersegmentasi
mse_median_roberts = hitung_mse(gray_roberts.astype(np.float32), sp_median_roberts.astype(np.float32))
mse_median_prewitt = hitung_mse(gray_prewitt.astype(np.float32), sp_median_prewitt.astype(np.float32))
mse_median_sobel = hitung_mse(gray_sobel.astype(np.float32), sp_median_sobel.astype(np.float32))
mse_median_frei = hitung_mse(gray_frei.astype(np.float32), sp_median_frei.astype(np.float32))

# 4. Grayscale tersegmentasi vs Salt & Pepper + Mean Filter tersegmentasi
mse_mean_roberts = hitung_mse(gray_roberts.astype(np.float32), sp_mean_roberts.astype(np.float32))
mse_mean_prewitt = hitung_mse(gray_prewitt.astype(np.float32), sp_mean_prewitt.astype(np.float32))
mse_mean_sobel = hitung_mse(gray_sobel.astype(np.float32), sp_mean_sobel.astype(np.float32))
mse_mean_frei = hitung_mse(gray_frei.astype(np.float32), sp_mean_frei.astype(np.float32))


# TAMPILKAN HASIL MSE DI CONSOLE
print("\n" + "=" * 90)
print("HASIL PERHITUNGAN MSE SEGMENTASI")
print("=" * 90)
print(f"Dimensi Gambar: {h} x {w} piksel")
print()
print("1. MSE: Grayscale tersegmentasi vs Salt & Pepper Noise tersegmentasi")
print(f"   Roberts   : {mse_sp_roberts:.2f}")
print(f"   Prewitt   : {mse_sp_prewitt:.2f}")
print(f"   Sobel     : {mse_sp_sobel:.2f}")
print(f"   Frei-Chen : {mse_sp_frei:.2f}")
print()
print("2. MSE: Grayscale tersegmentasi vs Gaussian Noise tersegmentasi")
print(f"   Roberts   : {mse_gauss_roberts:.2f}")
print(f"   Prewitt   : {mse_gauss_prewitt:.2f}")
print(f"   Sobel     : {mse_gauss_sobel:.2f}")
print(f"   Frei-Chen : {mse_gauss_frei:.2f}")
print()
print("3. MSE: Grayscale tersegmentasi vs S&P + Median Filter tersegmentasi")
print(f"   Roberts   : {mse_median_roberts:.2f}")
print(f"   Prewitt   : {mse_median_prewitt:.2f}")
print(f"   Sobel     : {mse_median_sobel:.2f}")
print(f"   Frei-Chen : {mse_median_frei:.2f}")
print()
print("4. MSE: Grayscale tersegmentasi vs S&P + Mean Filter tersegmentasi")
print(f"   Roberts   : {mse_mean_roberts:.2f}")
print(f"   Prewitt   : {mse_mean_prewitt:.2f}")
print(f"   Sobel     : {mse_mean_sobel:.2f}")
print(f"   Frei-Chen : {mse_mean_frei:.2f}")
print("=" * 90)


# VISUALISASI DIAGRAM BATANG
fig = plt.figure(figsize=(16, 10))

# Data untuk diagram batang
categories = ['S&P Noise', 'Gaussian Noise', 'S&P + Median', 'S&P + Mean']
x_pos = np.arange(len(categories))
bar_width = 0.2

roberts_values = [mse_sp_roberts, mse_gauss_roberts, mse_median_roberts, mse_mean_roberts]
prewitt_values = [mse_sp_prewitt, mse_gauss_prewitt, mse_median_prewitt, mse_mean_prewitt]
sobel_values = [mse_sp_sobel, mse_gauss_sobel, mse_median_sobel, mse_mean_sobel]
frei_values = [mse_sp_frei, mse_gauss_frei, mse_median_frei, mse_mean_frei]

# Buat diagram batang dengan 4 kelompok metode
bars1 = plt.bar(x_pos - 1.5*bar_width, roberts_values, bar_width, 
                label='Roberts', color='#FF6B6B', alpha=0.9, edgecolor='black', linewidth=1.5)
bars2 = plt.bar(x_pos - 0.5*bar_width, prewitt_values, bar_width, 
                label='Prewitt', color='#4ECDC4', alpha=0.9, edgecolor='black', linewidth=1.5)
bars3 = plt.bar(x_pos + 0.5*bar_width, sobel_values, bar_width, 
                label='Sobel', color='#FFE66D', alpha=0.9, edgecolor='black', linewidth=1.5)
bars4 = plt.bar(x_pos + 1.5*bar_width, frei_values, bar_width, 
                label='Frei-Chen', color='#95E1D3', alpha=0.9, edgecolor='black', linewidth=1.5)

# Tambahkan nilai MSE di atas setiap batang
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

add_values(bars1)
add_values(bars2)
add_values(bars3)
add_values(bars4)

# Styling
plt.xlabel('Jenis Pemrosesan Citra', fontsize=14, fontweight='bold')
plt.ylabel('Nilai MSE (Mean Square Error)', fontsize=14, fontweight='bold')
plt.title('Perbandingan MSE Segmentasi\n(Grayscale Tersegmentasi vs Berbagai Kondisi)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(x_pos, categories, fontsize=11, fontweight='bold')
plt.legend(loc='upper left', fontsize=11, framealpha=0.9, bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.gca().set_axisbelow(True)

plt.tight_layout()
plt.show()
