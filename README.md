# ML2025_Final_Project

---

#  Dino-AI: 自動學會玩 Google 小恐龍的機器學習專題

*(Dino-AI: Machine Learning Agent for the Google Chrome Dino Game)*

##  專案簡介 (Overview)

本專題目標是讓電腦**自動學會玩 Google Chrome 的離線恐龍遊戲**，
不使用任何手刻規則（rule-based），而是**透過觀察人類的示範畫面與操作紀錄進行學習**。

最終，AI 能根據螢幕畫面判斷何時跳躍、蹲下或保持原地，
在遊戲速度逐漸加快的情況下，自主做出即時反應並成功避開障礙。

---

##  專案概念 (Core Idea)

這是一個典型的「**行為模仿學習 (Behavior Cloning)**」應用。
我們先錄製人類玩家的遊戲過程（包含畫面與按鍵），再讓模型學習這些對應關係。

| 步驟          | 說明                               |
| ----------- | -------------------------------- |
|  **資料收集** | 擷取遊戲畫面並記錄人類玩家的按鍵動作     |
|  **特徵學習** | 將畫面轉成灰階影像，堆疊連續幀讓模型看出移動趨勢         |
|  **模型訓練** | 使用 CNN 模型學習「當畫面呈現這樣的情況時，人會做什麼動作」 |
|  **自動遊玩** | 以訓練好的模型即時預測動作，控制恐龍跳躍或閃避障礙        |

---

## 使用技術 (Tech Stack)

* **Python 3.10**
* **OpenCV** – 擷取與前處理遊戲畫面
* **PyTorch** – 建立與訓練卷積神經網路 (CNN)
* **pynput / pyautogui** – 模擬鍵盤輸入控制遊戲

##  專案目標 (Expected Results)

* 模型能根據畫面自動判斷 **何時跳躍、何時蹲下**。
* 當遊戲速度變快時，模型會自然學會「**提早起跳**」的行為。
* 不需要任何手刻規則或特徵判斷（完全基於資料學習）。

---

##  預期成果與應用 (Expected Outcomes)

* 展示 **AI 如何透過觀察學習人類行為**。
* 建立一個可延伸的遊戲自動化框架，可套用至其他簡易 2D 遊戲。
* 作為進一步研究強化學習 (Reinforcement Learning) 的基礎。

## breakdown

<img width="5044" height="1248" alt="image" src="https://github.com/user-attachments/assets/e3c82b85-92ba-42f9-bcfe-f3d79202d83f" />

## architecture
<img width="4496" height="1600" alt="image" src="https://github.com/user-attachments/assets/49ab0a71-f69c-402c-ad5d-64eb56183f2e" />

## api

| 檔案名稱 | 1. 輸入 (Input) | 2. 輸出 (Output) | 3. 主要參數 (Param) | 4. 方法/邏輯 (Method) | 5. 呼叫例子 (Call Example) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`main.py`**<br>(遊戲環境 Environment) | **動作指令 (Action)**<br>`0`: RUN<br>`1`: JUMP | **狀態與回饋 (Transition)**<br>`next_state` (List[float])<br>`reward` (float)<br>`done` (bool) | `FPS=60`<br>`skip_frames=4` (動作重複)<br>`width=600` | **`play_frame()`**: 執行單幀物理運算（移動恐龍、障礙物碰撞）。<br>**`step(action)`**: 執行動作並重複 4 幀，計算相對距離與獎勵。<br>**`get_state()`**: 將速度與距離正規化 (0~1)。 | `s_, r, done = env.step(action)`<br>`env.render()` |
| **`agent.py`**<br>(代理人/大腦 Agent) | **當前狀態 (State)**<br>或 **經驗樣本 (Experience)** | **動作索引 (Action Index)**<br>或 **更新權重** | `BATCH_SIZE=256`<br>`lr=0.0005` (微調時更低)<br>`epsilon` (探索率)<br>`decay=0.995` | **`act(state)`**: 依 Epsilon-Greedy 策略決定動作。<br>**`learn(experiences)`**: 從 Buffer 隨機抽樣並計算 Loss 更新網路。<br>**`update_epsilon()`**: 每個 Episode 結束後衰減探索率。 | `action = agent.act(state)`<br>`agent.step(s, a, r, s_, d)` |
| **`train.py`**<br>(訓練迴圈 Training Loop) | **環境物件 (Env)**<br>**代理人物件 (Agent)** | **權重檔案 (.pth)**<br>**訓練紀錄 (.csv)**<br>**Console Logs** | `num_episodes=10000`<br>`max_steps=5000`<br>`save_path` | **`train()`**: 主迴圈，協調 Env 與 Agent 互動。<br>**儲存機制**: 當 `current_score > best_score` 時呼叫 `torch.save` 儲存模型。 | `python train.py` |
| **`play.py`**<br>(驗收/推論 Inference) | **權重檔案 (.pth)**<br>**環境狀態 (State)** | **遊戲畫面渲染**<br>**最終分數 (Score)** | `RULE_BASED_MODE`<br>(True: 無敵腳本 / False: AI)<br>`ckpt` (權重路徑) | **`play()`**: 載入模型並關閉梯度 (`no_grad`) 進行遊玩。<br>**`get_rule_based_action()`**: 使用物理公式 (`距離 < 速度 * 係數`) 進行無敵模式驗證。 | `python play.py` |

## model acrhitecture
<img width="2804" height="3444" alt="image" src="https://github.com/user-attachments/assets/bc752f33-e48c-4551-a778-83de39a6dc71" />




