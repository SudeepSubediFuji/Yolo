import cv2

# 画像の読み込み
base_name = "cardboardface_angle5"
directory= "./test_img/"
img = cv2.imread(f'{directory}{base_name}.jpg')
h, w, _ = img.shape

# 男性の顔データ (例: クラス, X中心, Y中心, 横幅, 縦幅)
# ※データ形式に合わせて適宜調整してください
x_center, y_center, box_w, box_h =  0.071582,0.099774,0.143164,0.199548

# 2 0.487206 0.507441 0.230121 0.431194
# 0 0.071582 0.099774 0.143164 0.199548

# ピクセル座標に変換
x1 = int((x_center - box_w / 2) * w)
y1 = int((y_center - box_h / 2) * h)
x2 = int((x_center + box_w / 2) * w)
y2 = int((y_center + box_h / 2) * h)

# 画像に赤い枠線を引いて保存
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
cv2.imwrite('output1.png', img)
