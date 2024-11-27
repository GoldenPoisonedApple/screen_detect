import cv2
import numpy as np
import pyautogui


# もっと色細かくしてからk-meansで分類して、近いやつはくっつけるとかで色々やってみる




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
		lower_bound = np.array([h_min, 80, 100])
		upper_bound = np.array([h_max, 255, 255])
		return cv2.inRange(hsv_img, lower_bound, upper_bound)

def hsv_to_rgb(hsv_color):
		"""
		HSV色をRGB色に変換する関数
		"""
		return tuple(cv2.cvtColor(np.array([[hsv_color]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0][0])

def hsv_to_bgr(hsv_color):
    """
    HSV色をBGR色に変換する関数
    """
    bgr_color = cv2.cvtColor(np.array([[hsv_color]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    # BGRカラーが整数のタプルとして返されることを確認 矩形描画で怒られる
    return tuple(int(c) for c in bgr_color)

def decrease_brightness(frame, factor=0.5):
    """
    画像の明度を下げる関数。
    factor: 明度の減少率 (0.0 - 1.0)。
            1.0は明度をそのまま、0.0は完全に黒にする。
    """
    # 画像の輝度を減少させるため、スカラー値を掛ける
    frame = frame * factor
    frame = np.clip(frame, 0, 255).astype(np.uint8)  # 明度が255を超えないように制限
    return frame

def detect_mono_color(hsv_img, h_step, black_background=True):
		"""
		指定された色相範囲のマスクを作成し、輪郭を検出して描画する。
		"""
		# 色相ごとにマスクを作成
		mask = create_hue_mask(hsv_img, h_step*HUE_STEP, (h_step+1)*HUE_STEP - 1)
		# 輪郭の検出
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# 小さな輪郭を除去（面積が500以下のものを除去）
		min_area = 500
		filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
  
		# 背景の選択
		if black_background:
				img_contours = np.zeros_like(hsv_img, dtype=np.uint8)
		else:
				img_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 二値画像をカラー画像に変換

		# 代表色をBGRに変換
		bgr_color = hsv_to_bgr(HUE_HSV[h_step])
		for cnt in filtered_contours:
				# 輪郭の面積を計算
				area = cv2.contourArea(cnt)
	
				# 外接矩形を描画
				# x, y, w, h = cv2.boundingRect(cnt)
				# cv2.rectangle(img_contours, (x, y), (x + w, y + h), bgr_color, 2)

				# 面積を輪郭の近くに表示
				M = cv2.moments(cnt)
				if M["m00"] != 0:
						cX = int(M["m10"] / M["m00"])
						cY = int(M["m01"] / M["m00"])
						cv2.putText(img_contours, f"{h_step}: {int(area)}", (cX, cY - 30), 
												cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
				# 輪郭を描画
				cv2.drawContours(img_contours, [cnt], -1, bgr_color, 2)
		return img_contours


def main():
		print("輪郭検出を開始します。終了するには 'q' を押してください。")

		while True:
				# 画像を読み込む
				img = capture_screen()
    
    
				# HSV変換
				hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
				result_img = decrease_brightness(img.copy())
				for i in range(HUE_SEGMENTS):
						# 単色検出
						mono_countor = detect_mono_color(hsv_img, i)
						result_img = cv2.add(result_img, mono_countor)


				# 結果を表示
				cv2.imshow("Mask", result_img)

				# 'q'キーで終了
				if cv2.waitKey(1) & 0xFF == ord('q'):
						break

		# リソース解放
		cv2.destroyAllWindows()

if __name__ == "__main__":
		main()
