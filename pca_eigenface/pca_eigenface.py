import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people


def execute():
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for target, image, ax in zip(people.target, people.images, axes.ravel()):
        ax.imshow(image)
        ax.set_title(people.target_names[target])
    fig.savefig('pca_eigenface/pca_eigenface.png')

    print('people.images.shape: {}'.format(people.images.shape))
    print('Number of classes: {}'.format(len(people.target_names)))

    counts = np.bincount(people.target)
    for i, (count,name) in enumerate(zip(counts,people.target_names)):
        print("{0:25} {1:3}".format(name,count),end='  ')
        if(i + 1) % 3 == 0:
            print()

