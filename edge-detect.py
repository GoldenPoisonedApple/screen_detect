import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyautogui


def capture_screen():
    """スクリーンの左上1/4の範囲をキャプチャしてNumPy配列として返す"""
    screen_width, screen_height = pyautogui.size()
    region = (0, 0, screen_width // 2, screen_height // 2)  # 左上1/4範囲
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGBをBGRに変換
    return frame


def main():
    print("プレゼントボックス検出を開始します。終了するには 'q' を押してください。")

    while True:
        # 画像を読み込む
        img = capture_screen()
        
        # Cannyエッジ検出
        edges = cv2.Canny(img, threshold1=100, threshold2=200)
        
        # 結果を表示
        cv2.imshow("Red Mask", edges)  # 赤色のマスク
        
        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソース解放
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()