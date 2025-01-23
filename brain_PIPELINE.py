import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import skfuzzy as fuzz

def flush_memory():
    import gc
    gc.collect()

def sobel_edge_detection(img: np.ndarray) -> np.ndarray:
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    return sobel_edges

def kirsch_edge_detection(img: np.ndarray) -> np.ndarray:
    base_kernel = np.array([[-3, -3, 5],
                            [-3, 0, 5],
                            [-3, -3, 5]])
    
    kernels = [np.rot90(base_kernel, k=i) for i in range(8)][0]
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    responses = [cv2.filter2D(img_gray, -1, kernel) for kernel in kernels]
    
    kirsch_edges = np.max(responses, axis=0)
    
    return kirsch_edges

def canny_edge_detection(img: np.ndarray, th1: int = 100, th2: int = 200) -> np.ndarray:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(img_gray, (5, 5), 0)

    edges = cv2.Canny(blurred_img, th1, th2)
    return edges

def fuzzy_cmeans(img:np.ndarray, k=6, med_blur=1) -> np.ndarray:
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    pixel_vals = image.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) #,0.85
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))

    final_image = cv2.medianBlur(segmented_image, med_blur)

    return final_image

def morphological_operations(img: np.ndarray, operation = "open", kernel_size: int = 2) -> np.ndarray: # shortened method names are here
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if operation == 'dilate':
        result = cv2.dilate(img_gray, kernel, iterations=1)
    elif operation == 'erode':
        result = cv2.erode(img_gray, kernel, iterations=1)
    elif operation == 'open':
        result = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        result = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"Invalid operation: {operation}. Choose from 'dilate', 'erode', 'open', 'close'.")
    
    return result

def apply_segmentation(images:np.ndarray, op:str) -> np.ndarray: # flattened + normalized 
    if op == "SO":
        return np.apply_along_axis(sobel_edge_detection, 2, images)  # Sobel
    elif op == "CA":
        return np.apply_along_axis(canny_edge_detection, 2, images)  # Canny
    elif op == "FU":
        return np.apply_along_axis(fuzzy_cmeans, 2, images)  # Fuzzy C-means
    elif op == "MO":
        return np.apply_along_axis(morphological_operations, 2, images)  # morphological operations
    elif op == "MC":
        return np.apply_along_axis(lambda img: morphological_operations(img, "close", 2), 2, images)  # Close
    elif op == "MD":
        return np.apply_along_axis(lambda img: morphological_operations(img, "dilate", 2), 2, images)  # Dilate
    elif op == "OG":
        return images
    else:
        raise ValueError(f"Unsupported operation: {op}")

def normalize_images(images:np.ndarray) -> np.ndarray:
    return [img / 255.0 for img in images]

def extract_features(images):
    images = np.array(images)

    stacked_images = np.hstack([images])
    
    return images.reshape(images.shape[0], -1)

def train_and_evaluate(imsize, X_train, X_test, y_train, y_test, model_type="RandomForest", params=None,
                       seg_method="So",  
                       feature_method="FL"):

    if model_type == "RandomForest":
        model = RandomForestClassifier(**(params or {}))
    elif model_type == "SVC":
        model = SVC(**(params or {}))
    else:
        raise ValueError("Unsupported model type. Use 'RandomForest' or 'SVC'.")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    acc_rounded = round(acc, 2)

    imsize_str = f"{imsize[0]}x{imsize[1]}"
    file_prefix = f"cm_{imsize_str}_{seg_method}_{feature_method}_{model_type[:2]}_{acc_rounded}"

    cm_save_path = f"{file_prefix}.png"
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred).figure_.savefig(cm_save_path)

    report_save_path = f"{file_prefix}_report.txt"
    with open(report_save_path, "w") as f:
        f.write(classification_report(y_test, y_pred))
    
    print(f"Results saved: {cm_save_path}, {report_save_path}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred).figure_.savefig(f"cm_{imsize}_{model_type}.png")
    with open(f"{model_type}_report.txt", "w") as f:
        f.write(str(report))



if __name__ == "__main__":
    image_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
    segmentation_methods = ["SO", "OG", "CA"]
    feature_methods = ["fl"]
    models = ["RandomForest", "SVC"]

    y_train = np.load("train_test_y_images/y_train.npy")
    y_test = np.load("train_test_y_images/y_test.npy")

    for size in image_sizes:
        X_train = np.load(f"train_test_y_images/X_{size[0]}_train.npy")
        X_test = np.load(f"train_test_y_images/X_{size[0]}_test.npy")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")

    for method in segmentation_methods:
        segmented_image = apply_segmentation(X_train[0], method)
        plt.imshow(segmented_image, cmap="gray")
        plt.title(f"Segmentation Method: {method}, Image Size: {size}")
        plt.colorbar()
        plt.show()

        for seg_method in segmentation_methods:
            X_train_segmented = apply_segmentation(X_train, seg_method)
            X_test_segmented = apply_segmentation(X_test, seg_method)

            X_train_normalized = normalize_images(X_train_segmented)
            X_test_normalized = normalize_images(X_test_segmented)

            X_train_features = extract_features(X_train_normalized)
            X_test_features = extract_features(X_test_normalized)

            print(f"X_train shape after extraction: {X_train_features.shape}")
            print(f"X_test shape after extraction: {X_test_features.shape}")

            for model_type in models:
                train_and_evaluate(X_train_features, X_test_features, y_train, y_test, model_type)
                flush_memory()

