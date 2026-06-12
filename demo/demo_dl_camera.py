import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from collections import deque

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(SCRIPT_DIR, 'best_dl_model.keras')
    info_path = os.path.join(SCRIPT_DIR, 'dl_model_info.txt')
    
    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到模型檔案 '{model_path}'。請先執行 train_dl_models.py")
        sys.exit(1)
        
    model = tf.keras.models.load_model(model_path)
    
    model_type = 'MobileNetV3Small'
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            model_type = f.read().strip()
            
    print(f"✅ 成功載入深度學習模型: {model_type}")
    
    labels = ['Rock', 'Paper', 'Scissors']
    THRESHOLD = 0.75
    
    # 時間平滑：紀錄過去 N 幀的機率來取平均，讓辨識結果更穩定
    history_length = 3
    prob_history = deque(maxlen=history_length)
    
    camera_id = 0
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            pass

    cap = cv2.VideoCapture(camera_id)
    print(f"🎥 嘗試啟動攝影機 (ID: {camera_id})，按 'q' 鍵退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法讀取攝影機畫面")
            break

        h, w, _ = frame.shape
        size = 300
        x1, y1 = (w - size) // 2, (h - size) // 2
        x2, y2 = x1 + size, y1 + size

        # 繪製藍色 ROI 框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = frame[y1:y2, x1:x2]
        
        # 深度學習前處理
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (96, 96))
        
        if model_type == 'MobileNetV3Small':
            from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        else:
            from tensorflow.keras.applications.densenet import preprocess_input
            
        X = preprocess_input(np.expand_dims(resized, axis=0))
        
        # 進行預測
        probs = model.predict(X, verbose=0)[0]
        
        # 移動平均 (Temporal Smoothing)
        prob_history.append(probs)
        avg_probs = np.mean(prob_history, axis=0)
        
        max_prob = np.max(avg_probs)
        pred_idx = np.argmax(avg_probs)

        # 判斷結果或 Error
        if max_prob < THRESHOLD:
            result_text = f"Error (Conf: {max_prob:.2f})"
            color = (0, 0, 255)
        else:
            result_text = f"{labels[pred_idx]} ({max_prob:.2f})"
            color = (0, 255, 0)

        # 顯示在畫面上
        cv2.putText(frame, result_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Model: {model_type}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Deep Learning Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
