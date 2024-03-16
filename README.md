if You want to use your own image in this code  here is the update code for this task 



from matplotlib import pyplot as plt
# from sklearn.datasets import load_sample_image
import matplotlib.image as mpimg
# china=load_sample_image("china.jpg")
image_path = "china.jpg"  # Change this to the path of your image
china = mpimg.imread(image_path)

ax=plt.axes(xticks=[],yticks=[])
im = ax.imshow(china);
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets import load_sample_image

# china = load_sample_image("china.jpg")
import matplotlib.image as mpimg
# china=load_sample_image("china.jpg")
image_path = "china.jpg"  # Change this to the path of your image
china = mpimg.imread(image_path)
data = china / 255.0
height, width, channels = data.shape
data = data.reshape(height * width, channels)

def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(data[i, 0], data[i, 1], color=colors, marker='.')
    ax[0].set(xlabel='Channel 1', ylabel='Channel 2', xlim=(0, 1), ylim=(0, 1))
    ax[1].scatter(data[i, 0], data[i, 2], color=colors, marker='.')
    ax[1].set(xlabel='Channel 1', ylabel='Channel 3', xlim=(0, 1), ylim=(0, 1))
    plt.suptitle(title, size=20)

plot_pixels(data, "Pixel plot of China Image")
plt.show()

import warnings;warnings.simplefilter('ignore')
from sklearn.cluster import MiniBatchKMeans
kmeans=MiniBatchKMeans(16)
kmeans.fit(data)
new_colors=kmeans.cluster_centers_[kmeans.predict(data)]
plot_pixels(data,colors=new_colors,title="Reduced color space: 16 colors")
china_recolored=new_colors.reshape(china.shape)
fig, ax=plt.subplots(1,2,figsize= (16,6),subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image ',size=16)
ax[1].imshow(china_recolored)
ax[1].set_title("16-color Image", size=16);
