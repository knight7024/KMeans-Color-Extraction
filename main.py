import warnings
from ast import literal_eval
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

max_k = 5


def get_best_k(points):
    distortions = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(points)
        distortions.append(kmeans.inertia_)

    kn = KneeLocator(
        range(1, max_k + 1),
        distortions,
        curve='convex',
        direction='decreasing',
    )

    # Removable
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.axvline(x=kn.knee)
    plt.show()

    return kn


# 색상 추출
# Compared with Avery Colorbook from colornerd project (https://github.com/jpederson/colornerd)
def extract_colors(kmeans: KMeans, csv_name: str) -> list:
    extracted_colors = []
    data = pd.read_csv(csv_name, encoding='utf-8-sig')

    def color_distance(x: tuple, y: tuple) -> float:
        assert len(x) == 3 and len(y) == 3
        r_mean = (x[0] + y[0]) // 2
        r, g, b = abs(x[0] - y[0]), abs(x[1] - y[1]), abs(x[2] - y[2])
        # Colour Distance Algorithm (https://www.compuphase.com/cmetric.htm)
        return ((((512 + r_mean) * r * r) >> 8) + (4 * g * g) + (((767 - r_mean) * b * b) >> 8)) ** 0.5

    for color in kmeans.cluster_centers_:
        rgb_from = tuple(color[:3].astype("uint8"))

        min_dist = float('inf')
        final_color_name = ''
        for i in range(len(data)):
            row = data.loc[i]
            rgb_to, color_name = literal_eval(row[1]), row[0]

            dist = color_distance(rgb_from, rgb_to)
            if dist < min_dist:
                min_dist = dist
                final_color_name = color_name

        extracted_colors.append(final_color_name)
    return extracted_colors


def load_image(name):
    image = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    w, h, d = tuple(image.shape)
    # make image 2d-array
    image_array = np.reshape(image, (w * h, d))
    if d > 3:
        image_array = image_array[image_array[:, 3] != 0]

    return image_array


if __name__ == '__main__':
    image_array = load_image('463655_1_220_(1).png')

    kn = get_best_k(image_array)
    kmeans = KMeans(n_clusters=kn.knee, random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)

    extracted_colors = extract_colors(kmeans=kmeans, csv_name='avery_color_list.csv')
    print(extracted_colors)


    def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist


    hist = centroid_histogram(kmeans)


    def plot_colors(hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0

        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            # print(color[:3].astype("uint8"))
            startX = endX

        # return the bar chart
        return bar


    bar = plot_colors(hist, kmeans.cluster_centers_)

    # show our color bart
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
