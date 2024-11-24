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

def detect_white(frame):
    """白色の物体を検出して結果を描画"""
    # BGRをHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 白色のHSV範囲を指定
    lower_white = np.array([0, 0, 200])  # 白色の下限値
    upper_white = np.array([180, 60, 255])  # 白色の上限値

    # 範囲内の色をマスク
    mask = cv2.inRange(hsv, lower_white, upper_white)

    return mask



def main():
    print("色相を18分割して物体を抽出し、一画面に統合します。終了するには 'q' を押してください。")

    # 色相範囲に対応する代表的な色（BGR形式）
    hue_colors = [
        (0, 0, 255),       # 0~10 (赤)
        (0, 0, 204),       # 10~20 (赤オレンジ)
        (0, 128, 255),     # 20~30 (オレンジ)
        (0, 255, 255),     # 30~40 (オレンジ黄色)
        (0, 255, 204),     # 40~50 (黄色)
        (0, 255, 128),     # 50~60 (黄緑)
        (0, 255, 0),       # 60~70 (緑)
        (128, 255, 0),     # 70~80 (黄緑色)
        (255, 255, 0),     # 80~90 (青緑)
        (255, 255, 128),   # 90~100 (青)
        (255, 0, 0),       # 100~110 (青紫)
        (204, 0, 255),     # 110~120 (紫)
        (128, 0, 255),     # 120~130 (紫)
        (255, 0, 255),     # 130~140 (紫赤)
        (255, 0, 204),     # 140~150 (赤紫)
        (204, 0, 204),     # 150~160 (赤紫)
        (128, 0, 204),     # 160~170 (紫)
        (128, 0, 255)      # 170~179 (紫)
    ]
    
    while True:
        # 画像を読み込む
        img = capture_screen()
        
        # HSV変換
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 色相の範囲を18分割してマスクを作成
        result_img = np.zeros_like(img)  # 合成画像用の初期化（黒い画像）
        h_min = 0
        h_max = 10  # 10度ずつ分ける
        
        for i in range(18):
            # 色相範囲の下限と上限
            lower_bound = np.array([h_min, 100, 100])  # 色相範囲の下限
            upper_bound = np.array([h_max, 255, 255])  # 色相範囲の上限
            
            # マスクを作成
            mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
            
            # マスクされた部分に代表色を塗りつぶし
            color_mask = np.zeros_like(img)
            color_mask[mask == 255] = hue_colors[i]
            
            # 合成画像に加算
            result_img = cv2.add(result_img, color_mask)
            
            # 次の色相範囲に設定
            h_min = h_max
            h_max += 10
        
        # 合成された画像を表示
        cv2.imshow("Combined Hue Masks", result_img)

        # 白検出
        white_mask = detect_white(img)
        cv2.imshow("White Mask", white_mask)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソース解放
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
