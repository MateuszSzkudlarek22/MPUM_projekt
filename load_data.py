import os
import numpy as np
from get_features import extract_color_features, extract_texture_features, extract_shape_features
from sklearn.model_selection import train_test_split

def load_data(base_folder, val_size = 0.2, test_size = 0.2):
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

    for animal_label in animal_folders:
        animal_path = os.path.join(base_folder, animal_label)

        image_files = [f for f in os.listdir(animal_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = 0
        for img_file in image_files:
            print(count)
            count += 1

            #Pełen zbiór jest dość duży, jego wczytywanie trwa bardzo długo, na potrzeby testów
            #zmniejszyłem ją
            if count>100:
                break
            features = np.array([])
            img_path = os.path.join(animal_path, img_file)

            features = np.append(features, extract_color_features(img_path))
           # features = np.append(features, extract_texture_features(img_path))
           # features = np.append(features, extract_shape_features(img_path))

            X.append(features)
            y = np.append(y, label_map_1[animal_label])

    X = np.array(X)

    X_train , X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                test_size=test_size/(test_size+val_size), stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test
