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
# TODO: next, merge all pixels of the same word together to make one connected component using a morphologic operator
kernel = np.zeros((4,4),dtype=np.uint8)
kernel[1:3,:] = 1
plt.figure(figsize=figsize)
plt.imshow(kernel,cmap="gray")
plt.show()
dilated_im = cv2.dilate(im_th, kernel)
cv2.imwrite("output_SAVE3.png", dilated_im)
plt.figure(figsize=(20, 20))
plt.imshow(dilated_im, cmap='gray', vmin=0, vmax=255)
plt.show()

# %%


def find_words(dilated_im, im):
    res = im.copy()

    # TODO: draw rectengles around each word:
    # 1. find all connected components
    # 2. build a mask of only one connected component each time, and find it extremeties
    # TODO: did it came out perfect? Why? Why not?
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated_im)
    print(f'Detected {num_labels} words')
    return res

def show_connected_components(dilated_im, im):
    res = im.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated_im)
    print(f'Detected {num_labels} words')
    color_map = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]  # Setting background color to black
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
    return res_im


# %%
plt.figure(figsize=(20, 20))
plt.imshow(find_words(dilated_im, im))
plt.show()


# %%
# TODO: now we want to mark only the big title words, and do this ONLY using morphological operators


plt.figure(figsize=(20, 20))
plt.imshow(find_words(binary_only_title_cc_img, im))
plt.show()
