import os
import cv2
import numpy as np
import joblib
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_images_from_folder(folder_path):
    """讀取本地資料夾內的圖片並轉換為模型可用的陣列"""
    images = []
    labels = []
    # 定義標籤對應: 0=Rock, 1=Paper, 2=Scissors
    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    
    for category, label_idx in label_map.items():
        category_path = os.path.join(folder_path, category)
        
        # 處理多包一層資料夾的情況
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if subdirs:
                category_path = os.path.join(folder_path, subdirs[0], category)
        
        if not os.path.exists(category_path):
            print(f"⚠️ 警告: 找不到 {category} 的資料夾 -> {category_path}")
            continue
            
        print(f"📂 正在載入 {category} 的圖片...")
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (64, 64))
                    images.append(resized.flatten())
                    labels.append(label_idx)
                    
    return np.array(images) / 255.0, np.array(labels)

def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test):
    print(f"\n[{model_name}] 開始訓練...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"[{model_name}] 訓練完成，耗時: {train_time:.2f} 秒。開始評估...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    # 使用 weighted 平均來計算整體表現
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"--- {model_name} 評估結果 ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors'], zero_division=0))
    
    return model, acc, prec, rec, f1

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    print("=== 步驟 1: 讀取圖片 ===")
    X_train, y_train = load_images_from_folder(train_dir)
    X_test, y_test = load_images_from_folder(test_dir)

    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ 錯誤：找不到足夠的圖片，請檢查 dataset 資料夾。")
        return

    # 宣告三個模型：SVM 需開啟 probability=True，以便後續判斷 Error 手勢
    models = {
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }

    results = {}
    best_model_name = None
    best_model = None
    best_acc = 0.0

    print(f"\n📊 讀取完成！訓練樣本數: {len(X_train)}, 測試樣本數: {len(X_test)}")
    print("\n=== 步驟 2: 訓練與比較三個模型 ===")
    for name, model in models.items():
        trained_model, acc, prec, rec, f1 = train_and_evaluate(name, model, X_train, y_train, X_test, y_test)
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
        
        # 根據 Accuracy 決定最強模型
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = trained_model

    print("\n=== 步驟 3: 模型比較總結 ===")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['Accuracy']:.4f}     | {metrics['Precision']:.4f}     | {metrics['Recall']:.4f}     | {metrics['F1']:.4f}")

    print(f"\n🏆 表現最好的模型是: {best_model_name} (Accuracy: {best_acc:.4f})")

    # 儲存最好的模型到 demo 資料夾
    os.makedirs(demo_dir, exist_ok=True)
    model_path = os.path.join(demo_dir, 'best_rps_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"✅ 已將最好的模型 ({best_model_name}) 儲存至: {model_path}")
    print("💡 在 demo 時，如果模型輸出的最高機率偏低，我們會將其判定為 'Error'。")

if __name__ == "__main__":
    main()
