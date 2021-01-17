'''
 KNN - K nearest negihbors is one of the simplified supervised machine learning algorithm used for

 classifying a data point based on how its neighbors are classified.

 KNN stores all available cases and classifies new cases based on similiarity measure

 k in knn is a parameter that refers to thew number of nearest neigbhors to include in a majority voting process

 choosing k helps in increasing accuracy, chosing worng value of k results in wrong processing
 choosing value of k => take square root of total number of data points or take an odd value of k in even number of classified dataset

WE CAN USE KNN WHEN :
    data is labeled => dogs, cats, digits, alphabets
    data is noise free
    dataset is small

    knn is lazy learning algorithm
'''

#  DATA_DIR =


TEST_DATA_FILENAME = 't10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = 'train-labels.idx1-ubyte'


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')


def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number from mnist
        n_images = bytes_to_int(f.read(4))

        if n_max_images:
            n_images = n_max_images

        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))

        for image_idx in range(n_images):
            image = []

            for row_idx in range(n_rows):
                row = []

                for cold_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number from mnist
        n_labels = bytes_to_int(f.read(4))

        if n_max_labels:
            n_labels = n_max_labels

        for label_idx in range(n_labels):
            label = f.read(1)
            labels.append(label)
    return labels

# X_train for dataset
# y_train for labels
# total images is 60,000


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

# def extract_features_from_samples()


def extract_features(dataset):
    return [flatten_list(dataset) for sample in dataset]

# image pixels


def dist(x, y):
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
            for x_i, y_i in zip(x, y)
        ]) ** (0.5)


def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]


def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_sample in X_test:
        training_distances = get_training_distances_for_test_sample(
            X_train, test_sample)

        sorted_distances_indices = [
            pair[0]
            for pair in sorted(enumerate(
                training_distances,
                key=lambda x: x[1]))
        ]  # enumerate => breaking arrays
        print(sorted_distances_indices)
        y_sample = 5
        y_pred.append(y_sample)
    return y_pred


def main():
    X_train = read_images(TRAIN_DATA_FILENAME, 200)
    # print(len(X_train))
    # print(len(X_train[0]))
    # print(len(X_train[0][0]))
    y_train = read_labels(TRAIN_LABELS_FILENAME)
    X_test = read_images(TEST_DATA_FILENAME, 200)
    y_test = read_labels(TEST_LABELS_FILENAME)

    print(len(X_train[0]))
    print(len(X_test[0]))

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    knn(X_train, y_train, X_test, 3)

    print(len(X_train[0]))
    print(len(X_test[0]))

    # print(len(X_train))
    # print(len(y_train))
    # print(len(X_test))
    # print(len(y_test))


if __name__ == '__main__':
    main()
