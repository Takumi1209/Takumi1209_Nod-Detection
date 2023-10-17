import cv2
import numpy as np
import dlib

# 顔の特徴点を検出するモデルを読み込む
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# カメラから映像を取得する
cap = cv2.VideoCapture(0)

# 鼻の頂点の座標の初期値と頷きの回数を設定する
nose_tip_y = 0
nod_count = 0

# 鼻の頂点の座標の変化量の閾値を設定する
threshold = 4

while True:
    # フレームを読み込む
    ret, frame = cap.read()
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 顔を検出
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    # 顔ごとに処理
    for (x, y, w, h) in faces:
         # 顔の領域を切り出す
        face_roi = gray[y:y+h, x:x+w]

        landmarks = landmark_detector(gray, dlib.rectangle(x, y, x+w, y+h))

        # 鼻の頂点の座標を取得する (特徴点番号は30)
        x = landmarks.part(30).x
        y = landmarks.part(30).y

        # 鼻の頂点に赤い円を描く
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # 鼻の頂点の座標の初期値が設定されていない場合は設定する
        if nose_tip_y == 0:
            nose_tip_y = y

        # 鼻の頂点の座標の変化量を計算する
        diff = y - nose_tip_y

        # 鼻の頂点の座標が閾値以上下がった場合は頷きと判定し、回数を増やす
        if diff > threshold:
            nod_count += 1

            # 鼻の頂点の座標の初期値を更新する
            nose_tip_y = y

    # 頷きの回数を表示する
    cv2.putText(frame, f"Nod count: {nod_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # フレームを表示する
    cv2.imshow('Nod Detector', frame)
    # キー入力を待つ
    key = cv2.waitKey(1)
    # Escキーが押されたら終了
    if key == 27:
        break

# カメラとウィンドウを解放する
cap.release()
cv2.destroyAllWindows()
