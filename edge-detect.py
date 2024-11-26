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


def main():
    print("輪郭検出を開始します。終了するには 'q' を押してください。")

    while True:
        # 画像を読み込む
        img = capture_screen()

        # Cannyエッジ検出
        edges = cv2.Canny(img, threshold1=100, threshold2=250)

        # 輪郭検出
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 黒背景に輪郭を描画
        img_with_contours = np.zeros_like(img)  # 同じサイズで黒背景の画像を作成
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)  # 緑色で輪郭を描画

        # エッジ画像と輪郭画像を表示
        cv2.imshow("Edges", edges)  # エッジのみ表示
        cv2.imshow("Contours", img_with_contours)  # 黒背景に輪郭のみ表示

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソース解放
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
