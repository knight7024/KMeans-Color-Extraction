import warnings
from ast import literal_eval

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colour import delta_E, sRGB_to_XYZ, XYZ_to_Lab
from kneed import KneeLocator
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

color_name_list = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'red', 'purple', 'yellow', 'white']

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
def extract_colors(kmeans: KMeans, csv_name: str = None) -> list:
    ret_colors = []

    # data = pd.read_csv(csv_name, encoding='utf-8-sig')

    # def color_distance_RGB(x: tuple, y: tuple):
    #     assert len(x) == len(y) == 3
    #     r_mean = (x[0] + y[0]) // 2
    #     r, g, b = x[0] - y[0], x[1] - y[1], x[2] - y[2]
    #     # Colour Distance Algorithm (https://www.compuphase.com/cmetric.htm)
    #     return (((512 + r_mean) * r * r) >> 8) + (4 * g * g) + (((767 - r_mean) * b * b) >> 8)

    def color_distance_LAB(x: tuple, y: tuple):
        assert len(x) == len(y) == 3
        x_space = XYZ_to_Lab(sRGB_to_XYZ(np.array([x[0] / 255, x[1] / 255, x[2] / 255])))
        y_space = XYZ_to_Lab(sRGB_to_XYZ(np.array([y[0] / 255, y[1] / 255, y[2] / 255])))

        return delta_E(a=x_space, b=y_space, method='CIE 2000')

    for color in kmeans.cluster_centers_:
        rgb_from = tuple(color[:3].astype("uint8"))

        min_dist = float('inf')
        final_color_name = ''
        for rgb_to in color_names_dictionary:
            color_name = color_names_dictionary[rgb_to]

            dist = color_distance_LAB(rgb_from, rgb_to)
            if dist < min_dist:
                min_dist = dist
                final_color_name = color_name

        ret_colors.append(final_color_name)
    return ret_colors


def load_image(name):
    image = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    w, h, d = tuple(image.shape)
    # make image 2d-array
    image_array = np.reshape(image, (w * h, d))
    if d > 3:
        image_array = image_array[image_array[:, 3] != 0]

    return image_array


def init():
    global color_names_dictionary
    color_names_dictionary = {}
    for color_name in color_name_list:
        load_csv = pd.read_csv(f'colors/{color_name}_color_list.csv', encoding='utf-8-sig')
        for i in range(len(load_csv)):
            row = load_csv.loc[i]
            rgb = literal_eval(row[1])
            color_names_dictionary[rgb] = color_name


if __name__ == '__main__':
    init()
    image_array = load_image('463655_1_220_(1).png')

    kn = get_best_k(image_array)
    kmeans = KMeans(n_clusters=kn.knee if kn.knee else 1).fit(image_array)
    labels = kmeans.predict(image_array)

    extracted_colors = extract_colors(kmeans=kmeans)
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
            startX = endX

        # return the bar chart
        return bar


    bar = plot_colors(hist, kmeans.cluster_centers_)

    # show our color bart
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
