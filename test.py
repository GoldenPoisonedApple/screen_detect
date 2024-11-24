import cv2
import numpy as np
import pyautogui

def capture_screen():
    """スクリーンの左上1/4の範囲をキャプチャしてNumPy配列として返す"""
    screen_width, screen_height = pyautogui.size()
    region = (0, 0, screen_width // 2, screen_height // 2)  # 左上1/4範囲
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGBをBGRに変換
    return frame

def detect_blue_cubes(frame):
    """水色のキューブを検出して結果を描画"""
    # BGRをHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 水色のHSV範囲を指定 (調整が必要な場合があります)
    # lower_blue = np.array([90, 50, 50])  # 水色の下限値
    # upper_blue = np.array([130, 255, 255])  # 水色の上限値
    lower_blue = np.array([90, 76, 180])  # 水色の下限値
    upper_blue = np.array([100, 234, 242])  # 水色の上限値
    
    # 範囲内の色をマスク
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 輪郭を検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 面積が小さいノイズを除外
        if cv2.contourArea(contour) > 500:  # 面積閾値を調整
            # 外接矩形を描画
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame, mask

def main():
    print("水色のキューブ検出を開始します。終了するには 'q' を押してください。")

    while True:
        # スクリーンキャプチャ
        frame = capture_screen()

        # 水色のキューブを検出
        result_frame, mask = detect_blue_cubes(frame)

        # 結果を表示
        cv2.imshow("Detected Blue Cubes", result_frame)
        cv2.imshow("Mask", mask)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソース解放
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
