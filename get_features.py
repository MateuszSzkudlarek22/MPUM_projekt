import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

def extract_color_features(img_path, bins=32, resize_shape=(224, 224)):
    """
    function for extracting color features from photos
    :param img_path: path to image
    :param bins: number of bins in histograms
    :param resize_shape: size to which photos will be resized
    :return: vector of features related to colors
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize_shape)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    features = np.array([])

    for i, color_name in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([rgb_img], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features = np.append(features, hist)

    for i, color_name in enumerate(['h', 's']):
        hist = cv2.calcHist([hsv_img], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features = np.append(features, hist)

    for img_color in [rgb_img, hsv_img, lab_img]:
        for i in range(3):
            features = np.append(features, np.mean(img_color[:, :, i]))
            features = np.append(features, np.std(img_color[:, :, i]))

    return features

def extract_texture_features(img_path, resize_shape=(224, 224)):
    """
    function for extracting texture features from photos
    :param img_path: path to image
    :param resize_shape: size to which photos will be resized
    :return: vector of features related to colors
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize_shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG (Histogram of Oriented Gradients)
    hog_features, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )

    # LBP (Local Binary Pattern)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    # GLCM (Gray-Level Co-Occurrence Matrix)
    def extract_glcm_features(img, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):

        glcm = graycomatrix(img, distances, angles, 256, symmetric=True, normed=True)

        # Wyodrębnij różne właściwości GLCM
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()

        return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])

    gray_scaled = (gray // 32).astype(np.uint8)
    glcm_features = extract_glcm_features(gray_scaled)

    texture_features = np.concatenate([hog_features, lbp_hist, glcm_features])

    return texture_features


def extract_shape_features(img_path, resize_shape=(224, 224)):
    """
    function extracting features related to shape from photos

    Args:
        img_path: path to image
        resize_shape: size to which photos will be resized

    Returns:
        vector of shape features
    """
    # Wczytaj obraz
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize_shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Progowanie (można użyć bardziej zaawansowanych metod segmentacji)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Znajdź kontury
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []

    if contours:
        # Użyj największego konturu (zakładając, że to główny obiekt)
        contour = max(contours, key=cv2.contourArea)

        # Momenty konturu
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()

        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        features.extend(hu_moments)

        # Pole powierzchni i obwód
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        features.append(area)
        features.append(perimeter)

        # Współczynnik kształtu (circularity)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        features.append(circularity)

        # Prostokąt otaczający
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        extent = float(area) / (w * h) if w * h > 0 else 0
        features.append(aspect_ratio)
        features.append(extent)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        features.append(solidity)
    else:
        features = np.zeros(13)

    return np.array(features)