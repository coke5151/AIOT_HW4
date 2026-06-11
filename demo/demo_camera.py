import cv2
import joblib
import numpy as np
import os
import sys

def main():
    # 載入最強的模型
    model_path = 'best_rps_model.pkl'
    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到模型檔案 '{model_path}'。請先執行 train/train_models.py！")
        return
        
    print("⏳ 載入模型中...")
    clf = joblib.load(model_path)
    print("✅ 模型載入成功！\n")

    # 標籤對應表
    labels = ['Rock', 'Paper', 'Scissors']
    # 設定 Error 的判定門檻 (最高機率若小於此值，則判定為 Error)
    THRESHOLD = 0.55

    # 取得要使用的攝影機 ID (預設為 0)
    camera_id = 0
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print("⚠️ 警告：輸入的攝影機 ID 無效，將使用預設鏡頭 0")

    cap = cv2.VideoCapture(camera_id)
    print(f"🎥 嘗試啟動攝影機 (ID: {camera_id})，按 'q' 鍵退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 錯誤：無法讀取攝影機畫面。")
            break
            
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # === 新增：建立正方形的感測區域 (ROI) ===
        # 這樣才能避免原本長方形的畫面被壓扁成 64x64 導致手勢變形
        box_size = 300
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # 畫出藍色框框，提示使用者將手放在這裡
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(display_frame, "Put hand here (White background best)", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 擷取框內的影像
        roi = frame[y1:y2, x1:x2]
        
        # --- 影像前處理 (必須與訓練時完全一致) ---
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        X = (resized.flatten() / 255.0).reshape(1, -1)
        
        # --- 進行預測 ---
        probas = clf.predict_proba(X)[0]
        max_proba = np.max(probas)
        pred_idx = np.argmax(probas)
        
        # --- 判斷是否為 Error 手勢 ---
        if max_proba < THRESHOLD:
            gesture = "Error"
            color = (0, 0, 255) # 紅色
        else:
            gesture = labels[pred_idx]
            color = (0, 255, 0) # 綠色
            
        # 在畫面上顯示結果與信心度
        text = f"{gesture} ({max_proba*100:.1f}%)"
        cv2.putText(display_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, color, 3, cv2.LINE_AA)
        
        # 顯示畫面
        cv2.imshow("Gesture Recognition Demo", display_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
