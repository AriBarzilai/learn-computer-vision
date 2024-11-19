# %% [markdown]
# # EX2_2
# Find different words in newspaper article
# We'll do this using morphology operators and connected components.
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
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_02a_basic_image_processing/ex2/news.jpg"
    )
# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

figsize = (10, 10)

# %%
im = cv2.imread("news.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=figsize)
plt.imshow(im_gray, cmap="gray", vmin=0, vmax=255)
plt.show()

# %%
_, im_th = cv2.threshold(im_gray,140,255,cv2.THRESH_BINARY_INV)
plt.figure(figsize=(20, 20))
plt.imshow(im_th, cmap='gray', vmin=0, vmax=255)
plt.show()

# %%
def connect_word_letters(im_th, kernel):
    dilated_im = cv2.dilate(im_th, kernel)
    #cv2.imwrite("output_SAVE3.png", dilated_im)
    plt.figure(figsize=(20, 20))
    plt.imshow(dilated_im, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return dilated_im

kernel = np.zeros((4,4),dtype=np.uint8)
kernel[1:3,:] = 1
dilated_im = connect_word_letters(im_th, kernel)


# %%


def find_words(dilated_im, im):
    res = im.copy()
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(dilated_im)
    print(f'Detected {num_labels} words')
    for i in range(1, num_labels):  # start from 1 to skip background
        mask = labels == i
        res = plot_rec(mask, res)
    return res

def show_connected_components(dilated_im, im):
    # this function is not actually used, it's for fun
    res = im.copy()
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(dilated_im)
    print(f'Detected {num_labels} words')
    color_map = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]  # set background color to black
    output_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for label in range(num_labels):
        output_image[labels == label] = color_map[label]
    plt.figure(figsize=(20, 20))
    plt.imshow(output_image)
    plt.show()
    cv2.imwrite("output_CONNECTED.png", output_image)
    
    
show_connected_components(dilated_im, im)

def plot_rec(mask, res_im):
    # plot a rectengle around each word in res image using mask image of the word
    xy = np.nonzero(mask)
    y = xy[0]
    x = xy[1]
    left = x.min()
    right = x.max()
    up = y.min()
    down = y.max()

    res_im = cv2.rectangle(res_im, (left, up), (right, down), (0, 20, 200), 2)
    cv2.imwrite("output_BOXED.png", res_im)
    return res_im


# %%
plt.figure(figsize=(20, 20))
plt.imshow(find_words(dilated_im, im))
plt.show()


# %%
def split_regions(im_th):
    # splits the image into title regions and article regions
    # label == 0 is background/article, everything else is titles
    binary_only_title_cc_img = im_th.copy()
    kernel = np.ones((3,3),dtype=np.uint8)
    binary_only_title_cc_img = cv2.morphologyEx(binary_only_title_cc_img, cv2.MORPH_ERODE, kernel)
    kernel = np.ones((43,43),dtype=np.uint8)
    binary_only_title_cc_img = cv2.morphologyEx(binary_only_title_cc_img, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((15,15),dtype=np.uint8)
    binary_only_title_cc_img = cv2.morphologyEx(binary_only_title_cc_img, cv2.MORPH_OPEN, kernel, iterations=4)
    kernel = np.ones((30,30),dtype=np.uint8)
    binary_only_title_cc_img = cv2.morphologyEx(binary_only_title_cc_img, cv2.MORPH_CLOSE, kernel, iterations=20)

    plt.imshow(find_words(binary_only_title_cc_img, im))
    plt.show()
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary_only_title_cc_img)
    return num_labels, labels

num_labels, labels = split_regions(im_th)
article_mask = ((labels == 0).astype(np.uint8) * 255)
title_mask = ((labels > 0).astype(np.uint8) * 255)
# %%

title_region = cv2.bitwise_and(im_th, im_th, mask=title_mask)
article_region = cv2.bitwise_and(im_th, im_th, mask=article_mask)

kernel = np.ones((7,7),dtype=np.uint8)
title_res = connect_word_letters(title_region, kernel)

kernel = np.zeros((4,4),dtype=np.uint8)
kernel[1:3,:] = 1
article_res = connect_word_letters(article_region, kernel)

im_title = im.copy()
plt.figure(figsize=(20, 20))
title_final = find_words(title_res, im_title)
plt.imshow(title_final)
plt.show()
cv2.imwrite("output_TITLE.png", title_final)

im_article = im.copy()
plt.figure(figsize=(20, 20))
article_final = find_words(article_res, im_article)
plt.imshow(article_final)
plt.show()
cv2.imwrite("output_ARTICLE.png", article_final)

# %%

# %%
