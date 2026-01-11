
---

#  Dino-AI: 自動學會玩 Google 小恐龍的機器學習專題

*(Dino-AI: Machine Learning Agent for the Google Chrome Dino Game)*

##  專案簡介 (Overview)

本專題目標是讓電腦**自動學會玩 Google Chrome 的離線恐龍遊戲**，
不使用任何手刻規則（rule-based），而是**透過RL強化是學習進行訓練**。

最終，AI 能根據遊戲狀態資訊判斷何時跳躍、蹲下或保持原地，
在遊戲速度逐漸加快的情況下，自主做出即時反應並成功避開障礙。

---
##  專案概念 (Core Idea)

本專案旨在開發一個具備 **「自我學習能力」** 的自動遊玩機器人，使其能夠自主遊玩經典的 Chrome 恐龍跳躍遊戲 (Dino Run)。

* **核心邏輯**：不使用傳統的規則腳本 (Rule-based，如「距離小於 50 就跳」)，而是採用 **強化學習 (Reinforcement Learning)** 技術。
* **運作方式**：
1. **觀察 (Observe)**：AI 讀取遊戲當下的數據（如速度、障礙物距離）。
2. **決策 (Act)**：神經網路判斷當下應該「跑」、「跳」還是「蹲」。
3. **回饋 (Reward)**：環境根據結果給予獎懲（活著給正分，撞死給重罰）。
4. **學習 (Learn)**：透過試錯 (Trial-and-error) 與反向傳播算法，修正神經網路參數，追求最大化長期分數。

---

## 使用技術 (Tech Stack)

本專案採用 **輕量化且高效** 的技術堆疊，證明了在普通 CPU 上也能進行深度強化學習訓練。

* **程式語言**：Python 3.8
* **深度學習框架**：PyTorch
  - 用於建構神經網路 (`nn.Module`)、計算梯度 (`backward`) 與優化參數 (`Adam` Optimizer)。


* **遊戲環境開發**：Pygame
  - 客製化的遊戲引擎 (`main.py`)，負責物理運算、碰撞偵測以及將遊戲畫面轉換為數值狀態 (State Vector)。


* **核心演算法**：**DQN (Deep Q-Network)**
  - **Experience Replay (經驗重播)**：建立 `ReplayBuffer` 儲存過往經驗，打破資料相關性，穩定訓練。
  - **Target Network (目標網路)**：使用雙網路架構 (`local_nn` 與 `target_nn`)，並採用 **Soft Update** () 技術來穩定 Q 值目標。
  - **Epsilon-Greedy Strategy**：動態調整探索率，平衡「隨機探索」與「利用經驗」。


* **Q-Net 模型架構**：**MLP (多層感知機)**
* **輸入層**：6 個特徵 (速度、距離 1、距離 2、烏鴉高度、蹲下狀態、滯空狀態)。
* **隱藏層**：2 層全連接層 (128 neurons)，搭配 ReLU 激活函數。
* **輸出層**：3 個動作價值 (Run / Jump / Duck)。

---

##  專案目標 (Expected Results)

* 模型能根據畫面自動判斷 **何時跳躍、何時蹲下**。
* 當遊戲速度變快時，模型會自然學會「**提早起跳**」的行為。
* 不需要任何手刻規則（完全從0學習）。

---

##  預期成果與應用 (Expected Outcomes)

* 展示 **AI 如何透過重複遊玩逐步獲取高分行為**。
* 建立一個可延伸的遊戲自動化框架，可套用至其他簡易 2D 遊戲。
* 作為進一步研究強化學習 (Reinforcement Learning) 的基礎。

---

## Breakdown
<img width="5044" height="1768" alt="image" src="https://github.com/user-attachments/assets/c75f97c9-9ceb-4cc1-b940-2287a96ad901" />

#### 各方塊功能介紹：
##### 遊戲環境素材 (Game Environment)
`dino.py`：定義主角恐龍的動作（跑、跳、蹲）與生存狀態<br>
`cactus.py`：負責隨機生成並移動地面上的仙人掌障礙物<br>
`crows.py`：負責生成並控制不同高度的空中烏鴉障礙物<br>
`ground.py`：負責繪製並滾動地板，營造遊戲持續前進的視覺效果<br>
##### AI 訓練系統 (Training / Inference)
`main.py`：整合所有遊戲素材與物理運算的環境核心<br>
`agent.py`：負責決策動作並管理經驗記憶庫<br>
`train.py`：負責執行訓練迴圈，協調環境與 AI 互動並更新權重的訓練腳本<br>
`play.py`：載入訓練好的權重，不進行學習，單純展示 AI 最終成果的推論腳本<br>
`model.py`：定義神經網路架構<br>


## Architecture
<img width="4564" height="1764" alt="image" src="https://github.com/user-attachments/assets/bc25fa96-2363-4dd3-b88b-56156dea1356" />


## APIs

| 檔案名稱 | 1. 輸入 (Input) | 2. 輸出 (Output) | 3. 主要參數 (Param) | 4. 方法/邏輯 (Method) | 5. 呼叫例子 (Call Example) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`main.py`**<br>(遊戲環境 Environment) | **動作指令 (Action)**<br>`0`: RUN<br>`1`: JUMP | **狀態與回饋 (Transition)**<br>`next_state` (List[float])<br>`reward` (float)<br>`done` (bool) | `FPS=60`<br>`skip_frames=4` (動作重複)<br>`width=600` | **`play_frame()`**: 執行單幀物理運算（移動恐龍、障礙物碰撞）。<br>**`step(action)`**: 執行動作並重複 4 幀，計算相對距離與獎勵。<br>**`get_state()`**: 將速度與距離正規化 (0~1)。 | `s_, r, done = env.step(action)`<br>`env.render()` |
| **`agent.py`**<br>(代理人/大腦 Agent) | **當前狀態 (State)**<br>或 **經驗樣本 (Experience)** | **動作索引 (Action Index)**<br>或 **更新權重** | `BATCH_SIZE=256`<br>`lr=0.0005` (微調時更低)<br>`epsilon` (探索率)<br>`decay=0.995` | **`act(state)`**: 依 Epsilon-Greedy 策略決定動作。<br>**`learn(experiences)`**: 從 Buffer 隨機抽樣並計算 Loss 更新網路。<br>**`update_epsilon()`**: 每個 Episode 結束後衰減探索率。 | `action = agent.act(state)`<br>`agent.step(s, a, r, s_, d)` |
| **`train.py`**<br>(訓練迴圈 Training Loop) | **權重檔案 (.pth)**<br>**環境狀態 (State)** | **權重檔案 (.pth)**<br>**訓練紀錄 (.csv)**<br>**Console Logs** | `num_episodes=10000`<br>`max_steps=5000`<br>`save_path` | **`train()`**: 主迴圈，協調 Env 與 Agent 互動。<br>**儲存機制**: 當 `current_score > best_score` 時呼叫 `torch.save` 儲存模型。 | `python train.py` |
| **`play.py`**<br>(驗收/推論 Inference) | **權重檔案 (.pth)**<br>**環境狀態 (State)** | **遊戲畫面渲染**<br>**最終分數 (Score)** | `ckpt` (權重路徑) | **`play()`**: 載入模型並關閉梯度 (`no_grad`) 進行遊玩。<br> | `python play.py` |
---

## Q-Net 
<img width="2804" height="3532" alt="image" src="https://github.com/user-attachments/assets/9fbcc721-649a-4054-8e8e-a288f8412275" />


### 輸入狀態 :
(1) Speed   : 恐龍移動速度 <br>
(2) Dist1   : 與最近障礙物之距離 <br>
(3) Dist2   : 與最近第二個障礙物之距離 <br>
(4) Crow    : 烏鴉之高度 <br>
(5) Ducking : 目前是否蹲下 <br>
(6) In air  : 目前是否跳起 <br>
### 輸出動作 :
(1) Jump    : 跳躍 <br>
(2) Nothing : 與最近障礙物之距離 <br>
(3) Duck    : 蹲下 <br>

程式片段
```python
class Model(nn.Module):
    def __init__(self, state_size, action_size):
        super(Model, self).__init__()
        self.seed = torch.manual_seed(5)

        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))

        action = self.fc3(state)

        return action
```

---

## Reward 設計
### A. 活著就有分
每一幀畫面只要活著就 `+1` , 此為基本存活分數
### B. 無故跳躍扣分
跳躍即 `-1` , 此為避免無障礙物時也跳躍
### C. 死亡重罰
死亡時 `-100` , 此為最主要懲罰, 透過此懲罰在多次訓練下恐龍需學習避免死亡

程式片段:
```python

  # 推進一幀
  self.play_frame()

  # 計算獎勵
  reward_step = 1 # 活著就有分
  if action == 1:
      reward_step -= 1 # 跳躍消耗體力 (避免沒事亂跳)
  
  if self.playerDino.isDead:
      reward_step = -100 # 死亡重罰
      done = True
  
  self.t_reward += reward_step

```

---
## Loss Function 設計

為了訓練網路，我們需要定義一個損失函數 (Loss Function) 來衡量預測誤差。DQN 結合了 **Q-Learning** 與 **深度神經網路**，並引入了 **Target Network** 與 **Experience Replay** 來穩定訓練。

### 步驟 1: 計算目標值 (The Target)

首先，我們計算對於某個經驗樣本的「標籤」或「目標值」。為了避免網路自我震盪，DQN 使用一個獨立的 **Target Network** 來計算未來的最大價值。

對於第 $j$ 個經驗樣本，目標值 $y_j$ 計算如下：

$$
y_j = r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-)
$$

**符號定義：**
* $y_j$: 第 $j$ 次更新時使用的**目標值 (Target Value)**。
* $r_j$: 該步實際獲得的獎勵。
* $\phi_{j+1}$: 下一步的狀態特徵向量 (State Feature Vector)。在本專案中包含速度、障礙物距離、高度等 6 維特徵。
* $\hat{Q}$: **目標網路 (Target Network)**，結構與主網路相同，但參數不同，用於評估未來價值。
* $\theta^-$: 目標網路的參數。它是每隔特定週期從主網路參數 $\theta$ 複製過來的（凍結參數），用於提供穩定的訓練目標。

### 步驟 2: 損失函數 (The Loss Function)

我們希望主網路 $Q$ 的輸出能逼近上述的目標值 $y_j$。DQN 使用**均方誤差 (Mean Squared Error, MSE)** 作為損失函數：

$$
L_i(\theta_i) = \mathbb{E}_{(\phi, a, r, \phi') \sim U(D)} \left[ \left( \underbrace{y_i}_{\text{Target}} - \underbrace{Q(\phi, a; \theta_i)}_{\text{Prediction}} \right)^2 \right]
$$

將 $y_i$ 展開後的完整公式：

$$
L_i(\theta_i) = \mathbb{E}_{(\phi, a, r, \phi') \sim U(D)} \left[ \left( \underbrace{\left( r + \gamma \max_{a'} \hat{Q}(\phi', a'; \theta_i^-) \right)}_{\text{由 Target Network 計算}} - \underbrace{Q(\phi, a; \theta_i)}_{\text{由 Predict Network 計算}} \right)^2 \right]
$$

**符號定義：**
* $L_i(\theta_i)$: 第 $i$ 次迭代的損失值。
* $D$: **經驗回放池 (Replay Memory)**，儲存了過去的轉移經驗 $e = (\phi, a, r, \phi')$。
* $U(D)$: **均勻分佈 (Uniform Distribution)**。表示我們從經驗池 $D$ 中隨機抽樣 (Random Sample) 一個 Minibatch 來計算 Loss，以打破數據的相關性。
* $\theta_i$: 第 $i$ 次迭代時，主網路 (Predict Network) 的參數（這是我們透過梯度下降要更新的對象）。
* $\theta_i^-$: 第 $i$ 次迭代時，目標網路 (Target Network) 的參數（固定不變，直到下一次同步）。

---
## 成果展示
#### 運行play.py 執行推論
demo影片:


https://github.com/user-attachments/assets/7d55f5e9-4c49-434e-a283-624a7ee6098a

#### score與reward學習曲線
<img width="1200" height="1000" alt="Code_Generated_Image (2)" src="https://github.com/user-attachments/assets/e9a1e632-f9dd-43e5-a844-3bc0245e6c43" />






