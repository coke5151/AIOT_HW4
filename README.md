## 刷機

https://www.raspberrypi.com/software/

1.	下載 Raspberry Pi Imager
    ![alt text](image.png)
2.	選擇 Raspberry Pi 4 64-bit
    ![alt text](image-1.png)
3.	插上讀卡機
    ![alt text](image-2.png)
4.	輸入主機名，ssh 會用到
    ![alt text](image-3.png)
5.	首都 Taipei ;時區Asia/Taipei
    ![alt text](image-4.png)
6.	輸入用戶名及密碼，ssh 會用到
    ![alt text](image-5.png)
7.	開啟 ssh
    ![alt text](image-6.png)
8.	完成寫入
    ![alt text](image-7.png)

---

## 運行

https://github.com/BiBaIsAFish/RSP_demo

1.	下載 github
2.	Demo carema
    ![](17903.jpg)
    ![](17904.jpg)

---

## 評分標準

- 成功在 Raspberry Pi 4 上執行 test.py & carema.py 50%
- Demo 展示影片(carema) 15%
    - 從兩個模型中選擇較強的模型，寫一支程式將 carema 接收到的畫面接到模型上進行分類，並錄一段 demo 10 個手勢的短片
    - 執行 10 次手勢辨識，且須包含以下手勢
        - 石頭(Rock)
        - 剪刀(Scissors)
        - 布(Paper)
        - 其他錯誤手勢 (Error)
- 報告 35%
	- 需自行找兩個模型架構修改 20%
		- 至少需呈現 accuracy, precision, recall, F1-score
	- 需解釋更換模型原因及比較差異 15%

---

## 報告 (Part 3: Report)

### 1. 模型比較數據 (Accuracy, Precision, Recall, F1-Score)
為了解決傳統機器學習在處理原始影像像素時缺乏平移不變性（導致結果亂跳、無法準確辨識）的問題，我們改用卷積神經網路 (CNN) 進行遷移學習 (Transfer Learning)。我們將圖片縮放為 96x96 之 RGB 影像，並測試了以下兩種輕量級與經典的 CNN 模型：
1. **MobileNetV3Small** - 專為邊緣設備（如 Raspberry Pi）設計的輕量級神經網路。
2. **DenseNet121** - 具有密集連接特性的經典卷積神經網路。

實驗數據統整如下表所示：

| 模型 (Model) | Accuracy (準確率) | Precision (精確率) | Recall (召回率) | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV3Small** | **0.9462** | **0.9525** | **0.9462** | **0.9455** |
| DenseNet121 | 0.8306 | 0.8877 | 0.8306 | 0.8286 |

**🏆 最終選擇：MobileNetV3Small (Accuracy: 94.62%)** 表現最為優異，因此獲選作為攝影機即時推論的最終模型。

### 2. 解釋更換模型原因及比較差異
- **為何選擇這兩個模型：** 使用原本的 SVM、Random Forest，因為將影像展平為一維陣列輸入，完全喪失了圖片的 2D 空間特徵，導致推論極度不穩、誤判率高。因此我們改用 CNN 架構。我們選擇了 MobileNetV3Small 作為高效能輕量化模型的代表，以及 DenseNet121 作為傳統深層網路的對照組，兩者都不會抄襲範例的簡單 CNN 架構，且都有優秀的特徵萃取能力。
- **模型差異比較分析：**
  - **MobileNetV3Small (表現最佳)**：專門為終端設備設計，不僅參數量少，推論速度極快，而且在我們的資料集上收斂非常順利。由於它具備良好的空間平移不變性 (Translation Invariance)，完美解決了先前模型「只要手移動位置就辨識錯誤」的問題。
  - **DenseNet121 (表現普通)**：雖然 DenseNet121 擁有更複雜的網路結構與特徵重用機制，但在我們僅有 2500 張左右的小資料集上，較容易發生 Overfitting，或者需要更多的 Epoch 才能達到與輕量模型相同的準確率。因此最終測試的 Accuracy 僅有 83.06%，不敵 MobileNetV3Small。

### 3. 「其他錯誤手勢 (Error)」判斷機制
由於訓練資料集僅有「石頭、剪刀、布」三種類別，為了滿足作業**「判斷 Error 手勢」**的要求：
在 `demo_dl_camera.py` 即時辨識時：
1. 模型 (預設經過 Softmax) 會輸出預測為石頭、剪刀、布的各別機率 (Probability)。
2. 我們設定了一個 **60% 的機率門檻 (Threshold)**。
3. 若攝影機捕捉到的畫面，經模型預測其最高機率仍 **小於 0.6 (60%)**，則系統會判定該畫面不屬於任何已知手勢，並在畫面上顯示為 **Error** (紅色字樣)。這完美解決了無關手勢或無手勢狀態的辨識需求。

### 4. AI 協作對話截圖
*(請將你與 Antigravity AI 對話討論更換為 CNN 架構、訓練 MobileNetV3Small 與解決畫面亂跳問題的相關截圖貼在此處)*

---

## 如何執行手勢辨識 Demo
1. 確保攝影機已連接至設備。
2. 啟動虛擬環境並確保已安裝 Tensorflow 及 OpenCV (`tensorflow`, `opencv-python`, `numpy`)。
3. 執行新的深度學習攝影機推論程式：
   ```bash
   python demo/demo_dl_camera.py
   ```
4. 對著鏡頭 (藍色框框內) 比出石頭、剪刀、布，或者其他無關的手勢來測試 Error 判定。按 `q` 鍵即可退出程式。
