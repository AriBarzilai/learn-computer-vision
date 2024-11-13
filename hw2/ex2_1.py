# %% [markdown]
# # EX2_1
# build dilate and erode functions
# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

figsize = (10, 10)

# %%

img = np.zeros((50, 50))
img[20:30, 20:30] = 1

plt.figure(figsize=figsize)
plt.imshow(img,cmap="gray")
plt.show()

# %%
kernel = np.zeros((5,5),dtype=np.uint8)
kernel[2,:] = 1
kernel[:,2] = 1


plt.figure(figsize=figsize)
plt.imshow(kernel,cmap="gray")
plt.show()

# %%

# %%
def my_dilate(img, kernel):
    res_img = np.zeros_like(img)
    kernel_x_radius = kernel.shape[0] // 2
    kernel_y_radius = kernel.shape[1] // 2
    threshold = 1
    
    for x in range(kernel_x_radius, img.shape[0] - kernel_x_radius):
        for y in range(kernel_y_radius, img.shape[1] - kernel_y_radius):
            sliding_window = img[x - kernel_x_radius : x + kernel_x_radius + 1, 
                                 y - kernel_y_radius : y + kernel_y_radius + 1]
            if np.sum(sliding_window[kernel == 1]) >= threshold:
                res_img[x, y] = 1
    return res_img
                

plt.figure(figsize=figsize)
plt.imshow(my_dilate(img,kernel),cmap="gray")
plt.show()

def is_same_images(img1, img2):
    diff = cv2.absdiff(img1, img2)
    return not np.any(diff)

# %%
if is_same_images(cv2.dilate(img, kernel), my_dilate(img, kernel)):
    print("cv2.dilate & my_dilate are the same!")
else: 
    print("try again...")

# %%
def my_erode(img,kernel):
    res_img = np.zeros_like(img)
    kernel_x_radius = kernel.shape[0] // 2
    kernel_y_radius = kernel.shape[1] // 2
    threshold = np.sum(kernel)
    
    for x in range(kernel_x_radius, img.shape[0] - kernel_x_radius):
        for y in range(kernel_y_radius, img.shape[1] - kernel_y_radius):
            sliding_window = img[x - kernel_x_radius : x + kernel_x_radius + 1, 
                                 y - kernel_y_radius : y + kernel_y_radius + 1]
            if np.sum(sliding_window[kernel == 1]) >= threshold:
                res_img[x, y] = 1
    return res_img

plt.figure(figsize=figsize)
plt.imshow(my_erode(img,kernel),cmap="gray")
plt.show()

# %%
# TODO: show that cv2.erode and my_erode are the same using absolute difference
if is_same_images(cv2.erode(img,kernel), my_erode(img,kernel)):
    print("cv2.erode & my_erode are the same!")
else: 
    print("try again...")

# %%
