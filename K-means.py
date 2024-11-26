import cv2
import numpy as np
import pyautogui
from sklearn.cluster import KMeans

# 設定値（定数として管理）
HUE_SEGMENTS = 18  # 色相範囲の分割数
HUE_STEP = 180 // HUE_SEGMENTS  # 色相の範囲ステップ
HUE_HSV = [
    (0, 255, 255), (10, 255, 255), (20, 255, 255), (30, 255, 255), (40, 255, 255),
    (50, 255, 255), (60, 255, 255), (70, 255, 255), (80, 255, 255), (90, 255, 255),
    (100, 255, 255), (110, 255, 255), (120, 255, 255), (130, 255, 255), (140, 255, 255),
    (150, 255, 255), (160, 255, 255), (170, 255, 255)
]
WHITE_HSV_RANGE = (np.array([0, 0, 200]), np.array([180, 100, 255]))  # 白色のHSV範囲


def capture_screen(region_ratio=0.5):
    """
    スクリーンの左上部分をキャプチャし、BGR画像として返す。
    region_ratio: スクリーンの幅と高さを縮小する比率 (0.5で左上1/4)
    """
    screen_width, screen_height = pyautogui.size()
    region = (0, 0, int(screen_width * region_ratio), int(screen_height * region_ratio))
    screenshot = pyautogui.screenshot(region=region)
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return frame


def create_hue_mask(hsv_img, h_min, h_max):
    """
    指定された色相範囲に基づいてマスクを作成。
    """
    lower_bound = np.array([h_min, 100, 100])
    upper_bound = np.array([h_max, 255, 255])
    return cv2.inRange(hsv_img, lower_bound, upper_bound)


def detect_white(hsv_img):
    """
    白色の領域を検出してマスクを返す。
    """
    lower_white, upper_white = WHITE_HSV_RANGE
    return cv2.inRange(hsv_img, lower_white, upper_white)


def overlay_color_masks(hsv_img):
    """
    HSV画像に基づいて色相ごとのマスクを作成し、結果を合成して返す。
    """
    result_img = np.zeros_like(hsv_img, dtype=np.uint8)
    h_min = 0

    for hsv in HUE_HSV:
        h_max = h_min + HUE_STEP
        mask = create_hue_mask(hsv_img, h_min, h_max)
        color_mask = np.zeros_like(result_img)
        
        # HSVをRGBに変換して適用
        rgb_color = tuple(cv2.cvtColor(np.array([[hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0])
        color_mask[mask == 255] = rgb_color
        result_img = cv2.add(result_img, color_mask)
        h_min = h_max

    return result_img


def combine_images(color_img, white_mask):
    """
    色相の結果と白色マスクを重ねた画像を返す。
    """
    white_area = np.zeros_like(color_img)
    white_area[white_mask == 255] = (255, 255, 255)
    return np.where(white_area == (255, 255, 255), white_area, color_img)


def find_contours_and_cluster(frame, hsv_img):
    """
    エッジ検出とクラスタリングを行い、物体を分離して矩形で囲む
    """
    # Cannyエッジ検出
    edges = cv2.Canny(frame, threshold1=100, threshold2=250)

    # エッジ画像から輪郭を検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭情報をHSV色空間の値とともに保存
    hsv_values = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            hsv_values.append(hsv_img[y, x])  # hsv_imgはy,x順

    # KMeansクラスタリング
    kmeans = KMeans(n_clusters=3)  # クラス数を3に設定（クラスタ数は必要に応じて調整）
    hsv_values = np.array(hsv_values)
    kmeans.fit(hsv_values)
    
    # クラスタごとの色を取得
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    return contours, labels, cluster_centers


def main():
    print("色情報とエッジ情報を使って物体を分離し、クラスタリングします。終了するには 'q' を押してください。")

    while True:
        # スクリーンをキャプチャ
        frame = capture_screen()

        # HSV変換
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 色相マスクの作成と合成
        color_result = overlay_color_masks(hsv_img)

        # 白色検出と合成
        white_mask = detect_white(hsv_img)
        combined_img = combine_images(color_result, white_mask)

        # エッジ検出とクラスタリング
        contours, labels, cluster_centers = find_contours_and_cluster(frame, hsv_img)

        # クラスタの中心を使って色を決定（HSV -> BGR変換）
        for i, contour in enumerate(contours):
            # 各輪郭に対して矩形を描画
            x, y, w, h = cv2.boundingRect(contour)
            color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[cluster_centers[labels[i]]]]), cv2.COLOR_HSV2BGR)[0][0])
            cv2.rectangle(combined_img, (x, y), (x + w, y + h), color, 2)

        # 結果を表示
        cv2.imshow("Clustering and Object Detection", combined_img)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソース解放
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
