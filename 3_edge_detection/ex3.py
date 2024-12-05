# %%
# to run in google colab
import sys

if "google.colab" in sys.modules:

    def download_from_web(url):
        import requests

        response = requests.get(url)
        if response.status_code == 200:
            with open(url.split("/")[-1], "wb") as file:
                file.write(response.content)
        else:
            raise Exception(
                f"Failed to download the image. Status code: {response.status_code}"
            )

    download_from_web(
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_03_edge_detection/ex3/butterfly_noisy.jpg"
    )

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

figsize = (10, 10)


# %%
def bilateral_one_pixel(source, x, y, d, sigma_r, sigma_s):
    # === init vars
    filtered_pix = 0
    Wp = 0
    bottom_bound = source.shape[0]
    right_bound = source.shape[1]
    nom_sum = 0
    denom_sum = 0 

    # calculate filtered pixel for the given d-sized kernel
    for j in range(-d // 2, d // 2):
        for i in range(-d // 2, d // 2):
           if (y+j) < 0 or (x+i) < 0 or (y+j) >= bottom_bound or (x+i) >= right_bound:
               continue
           g_s = np.exp(-1 * (i**2 + j**2) / (2 * sigma_s **2)) 
           f_r = np.exp(-1 * (source[y,x] - source[y+j,x+i])**2 / (2 * sigma_r **2)) 
           w_p_xyij = g_s * f_r
           nom_sum += source[y+j,x+i]*w_p_xyij
           denom_sum += w_p_xyij
                   
    if denom_sum > 0:
        filtered_pix = nom_sum / denom_sum
        
    # make result uint8
    filtered_pix = np.clip(filtered_pix, 0, 255).astype(np.uint8)
    return filtered_pix


# %%
def bilateral_filter(source, d, sigma_r, sigma_s):
    # build empty filtered_image
    filtered_image = np.zeros(source.shape, np.uint8)
    # make input float
    source = source.astype(float)
    # d must be odd!
    assert d % 2 == 1, "d input must be odd"

    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            filtered_image[y, x] = bilateral_one_pixel(source, x, y, d, sigma_r, sigma_s)

    return filtered_image


# %%
# upload noisy image
src = cv2.imread("butterfly_noisy.jpg")
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10, 10))
plt.imshow(src, cmap="gray", vmin=0, vmax=255)
plt.colorbar()
plt.show()

# %%
# ======== run
d = 5  # edge size of neighborhood perimeter
sigma_r = 12  # sigma range
sigma_s = 16  # sigma spatial

my_bilateral_filtered_image = bilateral_filter(src, d, sigma_r, sigma_s)

plt.figure(figsize=(10, 10))
plt.imshow(my_bilateral_filtered_image)
plt.colorbar()
plt.show()

# %%
# compare to opencv
cv2_bilateral_filtered_image = cv2.bilateralFilter(src, d, sigma_r, sigma_s)

plt.figure(figsize=(10, 10))
plt.imshow(cv2_bilateral_filtered_image)
plt.colorbar()
plt.show()

# %%
# compare to regular gaussian blur
gaussian_filtered_image = cv2.GaussianBlur(src, (d, d), sigma_s)
plt.figure(figsize=(10, 10))
plt.imshow(gaussian_filtered_image)
plt.colorbar()
plt.show()

# %%
# copare canny results between regular  two images
th_low = 100
th_high = 200
res = cv2.Canny(my_bilateral_filtered_image, th_low, th_high)
plt.figure(figsize=(10, 10))
plt.imshow(res)
plt.colorbar()
plt.show()

res = cv2.Canny(gaussian_filtered_image, th_low, th_high)
plt.figure(figsize=(10, 10))
plt.imshow(res)
plt.colorbar()
plt.show()
