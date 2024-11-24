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

def detect_gift_box(frame):
    """緑の箱と赤いリボンを組み合わせたプレゼントボックスを検出"""
    # BGRをHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 緑色のHSV範囲を指定
    lower_green = np.array([40, 50, 50])  # 緑色の下限値
    upper_green = np.array([80, 255, 255])  # 緑色の上限値

    # 赤色のHSV範囲を指定（赤は2つの範囲に分かれる）
    lower_red1 = np.array([0, 50, 50])  # 赤色の下限値1
    upper_red1 = np.array([10, 255, 255])  # 赤色の上限値1
    lower_red2 = np.array([170, 50, 50])  # 赤色の下限値2
    upper_red2 = np.array([180, 255, 255])  # 赤色の上限値2

    # 緑色と赤色のマスクを作成
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 緑色の領域を検出
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 検出結果を描画
    result_frame = frame.copy()

    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > 500:  # 面積閾値を調整
            x, y, w, h = cv2.boundingRect(contour)

            # 緑色の領域内に赤色の領域があるかを確認
            roi_red_mask = red_mask[y:y+h, x:x+w]
            red_area = cv2.countNonZero(roi_red_mask)

            if red_area > 100:  # 赤い部分の面積閾値を調整
                # プレゼントボックスとみなして矩形を描画
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result_frame, "Gift Box", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result_frame, green_mask, red_mask

def main():
    print("プレゼントボックス検出を開始します。終了するには 'q' を押してください。")

    while True:
        # スクリーンキャプチャ
        frame = capture_screen()

        # プレゼントボックスを検出
        result_frame, green_mask, red_mask = detect_gift_box(frame)

        # 結果を表示
        cv2.imshow("Gift Box Detection", result_frame)  # 検出結果
        cv2.imshow("Green Mask", green_mask)  # 緑色のマスク
        cv2.imshow("Red Mask", red_mask)  # 赤色のマスク

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソース解放
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
