import cv2
import numpy as np
import pyautogui

# 設定値（定数として管理）
HUE_SEGMENTS = 18  # 色相範囲の分割数
HUE_STEP = 180 // HUE_SEGMENTS  # 色相の範囲ステップ
HUE_COLORS = [
    (0, 0, 255), (0, 0, 204), (0, 128, 255), (0, 255, 255), (0, 255, 204),
    (0, 255, 128), (0, 255, 0), (128, 255, 0), (255, 255, 0), (255, 255, 128),
    (255, 0, 0), (204, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 204),
    (204, 0, 204), (128, 0, 204), (128, 0, 255)
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

    for i, color in enumerate(HUE_COLORS):
        h_max = h_min + HUE_STEP
        mask = create_hue_mask(hsv_img, h_min, h_max)
        color_mask = np.zeros_like(result_img)
        color_mask[mask == 255] = color
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


def main():
    print("色相を18分割して物体を抽出し、一画面に統合します。終了するには 'q' を押してください。")

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

        # 結果の表示
        cv2.imshow("Combined Image", combined_img)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソース解放
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
