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
    
    
				"""水色のキューブを検出して結果を描画"""
				# BGRをHSVに変換
				hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

				# 水色のHSV範囲を指定 (調整が必要な場合があります)
				lower_blue = np.array([90, 76, 180])  # 水色の下限値
				upper_blue = np.array([100, 234, 242])  # 水色の上限値

				# 範囲内の色をマスク
				mask = cv2.inRange(hsv, lower_blue, upper_blue)

				# 輪郭の検出
				contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

				# 小さな輪郭を除去（面積が500以下のものを除去）
				min_area = 500
				filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

				# 輪郭の描画
				img_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 二値画像をカラー画像に変換

				for cnt in filtered_contours:
						# 輪郭の面積を計算
						area = cv2.contourArea(cnt)
			
						# 外接矩形を描画
						x, y, w, h = cv2.boundingRect(cnt)
						cv2.rectangle(img_contours, (x, y), (x + w, y + h), (255, 0, 0), 2)

						# 面積を輪郭の近くに表示
						M = cv2.moments(cnt)
						if M["m00"] != 0:
								cX = int(M["m10"] / M["m00"])
								cY = int(M["m01"] / M["m00"])
								cv2.putText(img_contours, f"{area:.2f}", (cX, cY - 30), 
														cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

						# 輪郭を描画
						cv2.drawContours(img_contours, [cnt], -1, (255, 0, 0), 2)

				# 結果を表示
				cv2.imshow("Contours", img_contours)

				# 'q'キーで終了
				if cv2.waitKey(1) & 0xFF == ord('q'):
						break

		# リソース解放
		cv2.destroyAllWindows()

if __name__ == "__main__":
		main()
