import cv2
from skimage import io, color, feature
from skimage.transform import resize
import os
import numpy as np
from scipy.fftpack import dct
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split

X_global = None
y_global = None

def extract_color_layout_descriptor(image, grid_size=8, num_y_coef=12, num_cb_coef=6, num_cr_coef=6):
    """
    Ekstrahuje deskryptor Color Layout (CLD) z obrazu.
    """
    # Upewnij się, że obraz jest w formacie RGB
    if len(image.shape) < 3:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]

    # Konwersja z RGB do YCbCr
    image_ycbcr = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2YCrCb)

    # Podział obrazu na siatkę
    height, width = image_ycbcr.shape[:2]
    block_height = height // grid_size
    block_width = width // grid_size

    # Przygotowanie reprezentacji dla każdego kanału
    y_channel = np.zeros((grid_size, grid_size))
    cb_channel = np.zeros((grid_size, grid_size))
    cr_channel = np.zeros((grid_size, grid_size))

    # Dla każdego bloku oblicz średni kolor
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * block_height, (i + 1) * block_height
            x_start, x_end = j * block_width, (j + 1) * block_width

            block = image_ycbcr[y_start:y_end, x_start:x_end]

            # Średnie wartości kanałów w bloku
            y_channel[i, j] = np.mean(block[:, :, 0])
            cb_channel[i, j] = np.mean(block[:, :, 1])
            cr_channel[i, j] = np.mean(block[:, :, 2])

    # Zastosowanie DCT (Discrete Cosine Transform)
    dct_y = dct(dct(y_channel.T, norm='ortho').T, norm='ortho')
    dct_cb = dct(dct(cb_channel.T, norm='ortho').T, norm='ortho')
    dct_cr = dct(dct(cr_channel.T, norm='ortho').T, norm='ortho')

    # Zygzak-skanowanie i wybór najważniejszych współczynników DCT
    def zigzag_scan(matrix, n_coef):
        # Implementacja zygzak-skanowania
        rows, cols = matrix.shape
        solution = [[] for _ in range(rows + cols - 1)]

        for i in range(rows):
            for j in range(cols):
                sum = i + j
                if sum % 2 == 0:  # zygzak w górę
                    solution[sum].insert(0, matrix[i, j])
                else:  # zygzak w dół
                    solution[sum].append(matrix[i, j])

        # Spłaszcz i zwróć N pierwszych współczynników
        flattened = [item for sublist in solution for item in sublist]
        return np.array(flattened[:n_coef])

    # Wybierz współczynniki z zygzak-skanowania
    y_coef = zigzag_scan(dct_y, num_y_coef)
    cb_coef = zigzag_scan(dct_cb, num_cb_coef)
    cr_coef = zigzag_scan(dct_cr, num_cr_coef)

    # Połącz współczynniki w jeden deskryptor
    descriptor = np.concatenate((y_coef, cb_coef, cr_coef))

    return descriptor

def extract_features(image_path):
    # Load the image
    img = io.imread(image_path)

    # Resize to a standard size
    img_resized = resize(img, (100, 100))

    if len(img_resized.shape) > 2:
        # Check if image has 4 channels (RGBA)
        if img_resized.shape[2] == 4:
            # Convert RGBA to RGB by removing alpha channel
            img_resized = img_resized[:, :, :3]

        # Convert to grayscale for HOG
        img_gray = color.rgb2gray(img_resized)
    else:
        # It's already grayscale
        img_gray = img_resized
        # Create a 3-channel image from grayscale
        img_resized = np.stack((img_resized,) * 3, axis=-1)

    #Extract HOG (Histogram of Oriented Gradients) features
    #hog_features = feature.hog(img_gray,
    #                       orientations=8,
    #                       pixels_per_cell=(10, 10),
    #                       cells_per_block=(2, 2))
    #print(len(hog_features))
    color_features = []

    # Generate histograms for each color channel (RGB)
    for channel in range(3):
        histogram, _ = np.histogram(img_resized[:, :, channel], bins=32, range=(0, 1))
        color_features.extend(histogram)

    # Alternatively, use HSV color space for more robust color representation
    img_hsv = color.rgb2hsv(img_resized)
    for channel in range(3):
        histogram, _ = np.histogram(img_hsv[:, :, channel], bins=32, range=(0, 1))
        color_features.extend(histogram)
    descriptor = extract_color_layout_descriptor(img_resized)
    # Combine HOG and color features
    combined_features = np.concatenate((color_features, descriptor))

    return combined_features

def load_data(base_folder, val_size = 0.2, test_size = 0.2, safe=False):
    images = []
    labels = []
    label_map = {}
    label_map_1 = {}

    animal_folders = [f for f in os.listdir(base_folder)
                      if os.path.isdir(os.path.join(base_folder, f))]

    for i, animal in enumerate(sorted(animal_folders)):
        label_map[i] = animal
        label_map_1[animal] = i

    X = []
    y = np.array([])

    global X_global, y_global
    if X_global is None:
        for animal_label in animal_folders:
            animal_path = os.path.join(base_folder, animal_label)

            image_files = [f for f in os.listdir(animal_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = 0
            for img_file in image_files:
                if count >800:
                    break
                count += 1
                print(count)
                # Pełen zbiór jest dość duży, jego wczytywanie trwa bardzo długo, na potrzeby testów
                # zmniejszyłem ją
                img_path = os.path.join(animal_path, img_file)
                features = extract_features(img_path)

                X.append(features)
                y = np.append(y, label_map_1[animal_label])

        #pca = PCA(n_components=400)
        X = np.array(X)
        #X = pca.fit_transform(X)
        X_global = X
        y_global = y
    else:
        X = X_global
        y = y_global

    X_train , X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                test_size=test_size/(test_size+val_size), stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test
