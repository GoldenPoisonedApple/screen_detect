import tkinter as tk
from PIL import ImageGrab
import cv2
import numpy as np

# 定数
MAX_WIDTH = 1536
MAX_HEIGHT = 835
TOP_BER_HEIGHT = 30
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

def capture_window_area():
    # ウィンドウの位置とサイズを取得
    root.update_idletasks()
    x = root.winfo_rootx() # 0が最小値
    y = root.winfo_rooty() # 23が最小値
    width = root.winfo_width() # 1536が最大値
    height = root.winfo_height() # 824が最大値
    
    # ウィンドウの位置とサイズを調整
    x = x/MAX_WIDTH * WINDOW_WIDTH
    y = y/MAX_HEIGHT * WINDOW_HEIGHT - TOP_BER_HEIGHT
    width = width/MAX_WIDTH * WINDOW_WIDTH
    height = height/MAX_HEIGHT * WINDOW_HEIGHT
    
    # 座標情報を表示
    print(f"Window position and size: x={x}, y={y}, width={width}, height={height}")


    # スクリーンキャプチャ（指定された範囲）
    bbox = (x, y, x + width, y + height)
    screenshot = ImageGrab.grab(bbox)

    # 画像をBGR形式に変換
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # キャプチャ結果を保存
    cv2.imwrite("captured_window_area.png", frame)
    print(f"Captured area saved as 'captured_window_area.png': {bbox}")

# 半透明ウィンドウを作成
root = tk.Tk()
root.geometry("400x300+100+100")  # 初期サイズと位置を指定
root.attributes("-transparentcolor", "gray")  # 背景色を透明化
root.configure(bg="gray")  # 背景色を設定（透明化する色と一致させる）

# ボタンでキャプチャを実行
button = tk.Button(root, text="Capture This Window", command=capture_window_area)
button.pack(pady=20)

root.mainloop()
