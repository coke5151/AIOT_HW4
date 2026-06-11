import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time

def load_data(data_dir, img_size=(96, 96)):
    labels_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    images = []
    labels = []
    
    for label_name, label_idx in labels_map.items():
        folder = os.path.join(data_dir, label_name)
        if not os.path.exists(folder):
            continue
            
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Deep learning models need RGB
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, img_size)
                    images.append(resized)
                    labels.append(label_idx)
                    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y

def build_model(base_model_func, input_shape=(96, 96, 3), num_classes=3):
    # 載入預訓練模型，不包含頂部的全連接層
    base_model = base_model_func(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # 凍結預訓練層 (Transfer Learning) 加速訓練
    base_model.trainable = False
    
    # 加上自訂的全連接層
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model

def evaluate_model(model_name, model, X_test, y_test, preprocess_func):
    print(f"\n[{model_name}] 開始評估...")
    X_test_preprocessed = preprocess_func(X_test.copy())
    y_pred_prob = model.predict(X_test_preprocessed)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"--- {model_name} 評估結果 ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    return acc, prec, rec, f1

def main():
    print("=== 步驟 1: 讀取圖片 ===")
    train_dir = os.path.join('..', 'dataset', 'train')
    test_dir = os.path.join('..', 'dataset', 'test')
    
    # 使用 96x96 解析度 (在樹莓派上推理速度較快)
    IMG_SIZE = (96, 96)
    
    X_train_raw, y_train_raw = load_data(train_dir, IMG_SIZE)
    X_test_raw, y_test_raw = load_data(test_dir, IMG_SIZE)
    
    print(f"📊 訓練樣本數: {len(X_train_raw)}, 測試樣本數: {len(X_test_raw)}")
    
    # 轉換 Label 為 One-Hot Encoding
    y_train_cat = to_categorical(y_train_raw, num_classes=3)
    
    results = {}
    
    print("\n=== 步驟 2: 訓練 MobileNetV3Small ===")
    # MobileNetV3Small 的前處理：需要將像素縮放到 [0, 1] 或 [-1, 1] 嗎？
    # tf.keras.applications.MobileNetV3Small 內部有前處理邏輯，但直接給 0-255 RGB 陣列即可
    # 為了安全，我們手動縮放到 [0, 1] 或使用 preprocess_input
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mb_preprocess
    
    X_train_mb = mb_preprocess(X_train_raw.copy())
    
    mb_model, _ = build_model(MobileNetV3Small, input_shape=(96, 96, 3))
    start_time = time.time()
    mb_model.fit(X_train_mb, y_train_cat, epochs=5, batch_size=32, verbose=1)
    train_time = time.time() - start_time
    print(f"[{'MobileNetV3Small'}] 訓練完成，耗時: {train_time:.2f} 秒。")
    
    results['MobileNetV3Small'] = evaluate_model('MobileNetV3Small', mb_model, X_test_raw, y_test_raw, mb_preprocess)
    
    print("\n=== 步驟 3: 訓練 DenseNet121 ===")
    from tensorflow.keras.applications.densenet import preprocess_input as dn_preprocess
    X_train_dn = dn_preprocess(X_train_raw.copy())
    
    dn_model, _ = build_model(DenseNet121, input_shape=(96, 96, 3))
    start_time = time.time()
    dn_model.fit(X_train_dn, y_train_cat, epochs=5, batch_size=32, verbose=1)
    train_time = time.time() - start_time
    print(f"[{'DenseNet121'}] 訓練完成，耗時: {train_time:.2f} 秒。")
    
    results['DenseNet121'] = evaluate_model('DenseNet121', dn_model, X_test_raw, y_test_raw, dn_preprocess)
    
    print("\n=== 步驟 4: 模型比較總結 ===")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score'}")
    print("-" * 75)
    best_model_name = None
    best_acc = 0
    
    for name, (acc, prec, rec, f1) in results.items():
        print(f"{name:<25} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            
    print(f"\n🏆 表現最好的模型是: {best_model_name} (Accuracy: {best_acc:.4f})")
    
    best_model_path = os.path.join('..', 'demo', 'best_dl_model.keras')
    if best_model_name == 'MobileNetV3Small':
        mb_model.save(best_model_path)
    else:
        dn_model.save(best_model_path)
        
    print(f"✅ 已將最好的模型 ({best_model_name}) 儲存至: {best_model_path}")
    print("💡 為了紀錄預處理方式，我們也將模型名稱寫入一個文字檔：")
    with open(os.path.join('..', 'demo', 'dl_model_info.txt'), 'w') as f:
        f.write(best_model_name)

if __name__ == '__main__':
    main()
