import tkinter as tk
from PIL import ImageGrab
import cv2
import numpy as np

# 定数
MAX_WIDTH = 1536
MAX_HEIGHT = 830
TOP_BER_HEIGHT = 30
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

def capture_window_area(root):
	"""
		キャプチャ範囲指定ウィンドウの範囲をキャプチャして返す
	"""
	root.update_idletasks()
	x = root.winfo_rootx() # 0が最小値
	y = root.winfo_rooty() # 23が最小値
	width = root.winfo_width() # 1536が最大値
	height = root.winfo_height() # 824が最大値
	
	# ウィンドウの位置とサイズを調整
	# 何故か他のとこから参照すると変換が要らない
	# x = x/MAX_WIDTH * WINDOW_WIDTH
	# y = y/MAX_HEIGHT * WINDOW_HEIGHT
	# width = width/MAX_WIDTH * WINDOW_WIDTH
	# height = height/MAX_HEIGHT * WINDOW_HEIGHT - TOP_BER_HEIGHT
	
	# 座標情報を表示
	# print(f"Window position and size: x={x}, y={y}, width={width}, height={height}")


	# スクリーンキャプチャ（指定された範囲）
	bbox = (x, y, x + width, y + height)
	screenshot = ImageGrab.grab(bbox)

	# 画像をBGR形式に変換
	frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

	# 画像を1/2に縮小
	# frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)


	# キャプチャ結果を保存
	return frame


def init_capture_window():
	"""キャプチャ範囲指定ウィンドウ作成\n
	Returns:  キャプチャ範囲指定ウィンドウ
	"""
	root = tk.Tk("Capture Window")
	root.geometry("400x300+100+100")  # 初期サイズと位置を指定
	root.configure(bg="gray")  # 背景色を設定（透明化する色と一致させる）
	root.attributes("-transparentcolor", "gray")  # 背景色を透明化
	root.wm_attributes("-topmost", 1)  # 最前面に表示
	root.update() # ウィンドウの初期化

	return root

def main():
	print("範囲指定キャプチャを開始します。終了するには 'q' を押してください。")

	root = init_capture_window()

	while True:
		# スクリーンキャプチャ
		frame = capture_window_area(root)

		# 結果を表示
		cv2.imshow("Captured", frame)

		# 'q'キーで終了
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# リソース解放
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()