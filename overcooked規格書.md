# Overcooked AI Imitation Learning (IL)
<img width="380" height="214" alt="image" src="https://github.com/user-attachments/assets/8e107d0b-088f-4732-afd7-8e0e47f86573" />

利用畫面擷取 + CNN 模型訓練 AI 自主遊玩 Overcooked

本專案以 **行為模仿學習（Behavior Cloning）** 為核心方法，  
透過「畫面擷取 + 玩家控制器輸入」的方式蒐集資料，  
再使用 CNN 模型訓練一個能自主遊玩 Overcooked 的 AI。

目標是建立一套 **不需要修改遊戲、不需要 RL 環境**、  
即可讓 AI 學會從畫面推動作的輕量級方法。

---

# 1. 專案簡介

Overcooked 是一款合作型的高互動遊戲，  
傳統 RL 方法需要控制遊戲模擬環境、設計 reward、反覆訓練，成本極高。

本專案採用更實際的方式：

1. 人類玩家遊玩 Overcooked  
2. 同步記錄：
   - 螢幕畫面（Frame）
   - 鍵盤 / 控制器輸入（Action）
3. 訓練 CNN 模型讓 AI 學習 **畫面 → 動作** 的映射
4. 讓 AI 在遊戲中即時運作，模擬玩家行為

整體流程簡單、易於實作，非常適合作為課程或研究專題。

---

## 1. 系統需求（Requirements）

## 1.1 功能需求（Functional Requirements）

### **資料蒐集**
- 擷取固定 FPS 的遊戲畫面（建議 10–20 FPS）
- 同步紀錄玩家的鍵盤 / 手把按鍵
- 產生 `(frame, action)` 的訓練資料
- 提供圖像前處理（縮放、裁切、正規化）

### **模型訓練**
- 建立 CNN 動作分類模型
- 提供訓練、驗證、模型儲存功能
- 支援 GPU 加速（可選）

### **AI 部署**
- 使用即時畫面進行動作推論
- 自動將預測動作轉成鍵盤指令
- 可啟動「AI 自動遊玩模式」

---

## 1.2 效能需求（Performance Requirements）

### **訓練階段**
- 至少 10,000 張以上訓練資料
- 模型訓練準確率 ≥ 80%
- 模型推論速度 ≤ 50ms（20 FPS 以上）

### **AI 實際表現**
- 能在單人模式完成簡單關卡操作（如切菜、煮湯）
- AI 行為需連續且自然，不得不停 jitter
- AI 必須能處理小量的場景變動

---

## 1.3 介面需求（Interface Requirements）

### **外部介面（External Interfaces）**

#### **螢幕擷取介面**
- 選擇螢幕區域（bounding box）
- 回傳畫面：`RGB image (H, W, C)`

#### **動作輸出介面**
- 將模型輸出映射為鍵盤按鍵，例如：
  - 上：W  
  - 下：S  
  - 左：A  
  - 右：D  
  - 互動：Space  
- 允許 `config/action_map.json` 自訂按鍵配置

---

# 1.4. 限制（Limitations）

- **1. AI 上限取決於玩家示範的品質**  
- **2. Covariate Shift 問題：**  
  AI 落入未見過的狀態時可能會卡住  
- **3. 多人互動複雜度高**  
  單人模仿學習不適用於多人協作策略  
- **4. 缺乏長期規劃能力**  
  模型僅學習短期行為，不會自行規劃策略  
- **5. 場景依賴高**  
  不同場景、不同 UI 可能需要重蒐集資料再訓練  

---

# 1.5. 驗收標準（Acceptance Criteria）

### **資料收集階段**
- 每張 frame 都具有正確同步的 action label  
- 資料無損毀、FPS 維持穩定  
- 訓練資料格式通過自動檢查（resolution / label consistency）

### **模型訓練階段**
- 訓練準確率 ≥ 80%  
- 驗證準確率 ≥ 75%  
- Loss 曲線穩定下降、不震盪  

### **AI 遊玩階段**
- AI 能完成基本動作（拿取、切、煮、出餐）  
- AI 能連續遊玩至少 1 分鐘不中斷  
- 不得出現長時間卡住、原地轉圈等異常行為  

---
