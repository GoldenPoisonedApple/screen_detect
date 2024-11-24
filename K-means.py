import cv2
import numpy as np
import pyautogui
from skimage.feature import hog
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 画面キャプチャを取得する関数
def capture_screen():
    """スクリーンの左上1/4の範囲をキャプチャしてNumPy配列として返す"""
    screen_width, screen_height = pyautogui.size()
    region = (0, 0, screen_width // 2, screen_height // 2)  # 左上1/4範囲
    screenshot = pyautogui.screenshot(region=region)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# 色ヒストグラム特徴量抽出
def extract_color_histogram(image):
    # BGRカラーで色ヒストグラムを計算
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

# HOG特徴量抽出
def extract_hog_features(image):
    # グレースケールに変換
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features

# CNN特徴量抽出（ResNet50を使用）
def extract_cnn_features(image_path):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    features = model.predict(img_array)
    return features.flatten()

# 複数の画像の特徴量をリストに格納
features_list = []

# 例えば、複数の画像（例えば画面キャプチャを何度も取る）に対して特徴量を抽出
for _ in range(10):  # 例として10回画面キャプチャ
    screen_image = capture_screen()
    color_histogram_features = extract_color_histogram(screen_image)
    hog_features = extract_hog_features(screen_image)
    features = np.concatenate([color_histogram_features, hog_features])
    features_list.append(features)

# 次元削減
pca = PCA(n_components=10)  # サンプル数に合わせて10に設定
reduced_features = pca.fit_transform(features_list)

# K-meansクラスタリング
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(reduced_features)

# クラスタリング結果の表示
labels = kmeans.labels_

# 可視化
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
plt.title("K-means Clustering")
plt.show()

# 結果の表示
print(f'クラスタラベル: {labels}')

