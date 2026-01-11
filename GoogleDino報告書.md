
---

#  Dino-AI: 自動學會玩 Google 小恐龍的機器學習專題

*(Dino-AI: Machine Learning Agent for the Google Chrome Dino Game)*
本報告分成三部分 <br>
> 一. 專案規格 <br>
> 二. 系統架構與實作 <br>
> 三. Deep Q-Network (DQN) 詳細理論 <br>


---
<br>

# 一. 專案規格 (Project Specification)

---

##  專案簡介 (Overview)

本專題目標是讓電腦**自動學會玩 Google Chrome 的離線恐龍遊戲**，
不使用任何手刻規則（rule-based），而是**透過RL強化是學習進行訓練**。

最終，AI 能根據遊戲狀態資訊判斷何時跳躍、蹲下或保持原地，
在遊戲速度逐漸加快的情況下，自主做出即時反應並成功避開障礙。

---
##  1. 功能需求 (Functional Requirements)

本系統需具備以下核心功能，以滿足強化學習代理人 (Agent) 在遊戲環境中的訓練與運作：

- **FR-01 遊戲環境模擬 (Game Simulation)**
  - 系統需基於 **Pygame** 建立恐龍遊戲環境。
  - 需包含物理引擎，處理重力、跳躍拋物線與地面滾動速度。
  - 需動態生成障礙物（不同寬高的仙人掌、不同高度的烏鴉）。
  - 需具備精確的 **碰撞偵測 (Collision Detection)** 機制，當恐龍接觸障礙物時判定遊戲結束。


- **FR-02 狀態特徵提取 (State Feature Extraction)**
  - 系統不使用原始圖像 (Pixel)，而是提取 **特徵向量 (Feature Vector)** 作為 AI 輸入。
  - 輸入向量需包含至少 6 個維度：`[遊戲速度, 與最近障礙物距離, 與第二近障礙物距離, 烏鴉高度, 是否蹲下, 是否滯空]`。
  - 所有數值需經過正規化 (Normalization) 處理至 0~1 之間。


- **FR-03 DQN 代理人核心 (DQN Agent Core)**
  - 需實作深度 Q 網路 (Deep Q-Network) 演算法。
  - 需包含 **經驗回放池 (Replay Buffer)** 以儲存 `(state, action, reward, next_state, done)` 樣本。
  - 需包含 **雙網路架構 (Target Network & Local Network)**，並實作 **軟更新 (Soft Update)** 機制。
  - 需實作 **Epsilon-Greedy** 策略，支援從高探索率逐漸衰減至低探索率。


- **FR-04 訓練監控與記錄 (Training Monitoring)**
  - 系統需在每個 Episode 結束後計算並記錄：總獎勵 (Total Reward)、遊戲分數 (Score)、平均 Loss、平均 Q 值 (Avg Q)。
  - 需將上述數據輸出為 CSV 檔案，並繪製成學習曲線圖表 (Learning Curve)。
  - 當分數創新高時，需自動儲存模型權重 (`.pth` 檔案)。


- **FR-05 推論模式 (Inference Mode)**
  - 需提供獨立的執行腳本 (`play.py`)。
  - 需能夠載入預訓練的權重檔，並在關閉探索 (Epsilon=0) 的狀態下進行遊戲展示。

---

## 2. 效能需求 (Performance Requirements)

本系統需在有限的計算資源下達到高效的訓練與運作：

* **PR-01 輕量硬體效能 (CPU Efficiency)**
  * 模型架構需輕量化，確保在無 GPU 加速的 **純 CPU 環境** 下，訓練速度能至少超過 **60 FPS** 。

* **PR-02 收斂速度 (Convergence Speed)**
  * 需在 **2000 個 Episodes** 內展現出明顯的學習趨勢（Score 與 Q-Value 趨勢顯著優於隨機操作）。

* **PR-04 推論延遲 (Inference Latency)**
  * 在 `play.py` 模式下，單次決策 (Forward Pass) 的延遲需低於 **16ms** (支援 60 FPS 刷新率)，確保操作無延遲感。

---

## 3. 限制條件 (Constraints)

本專案在開發與執行過程中受到以下條件限制：

* **C-01 輸入限制 (State-based Input)**
  * 受限於 CPU 算力與訓練效率考量，**禁止使用 CNN (卷積神經網路)** 直接處理遊戲截圖，必須使用手動提取的數值特徵。

* **C-02 動作空間限制 (Discrete Action Space)**
  * 動作空間固定為離散數值：`0 (Run)`, `1 (Jump)`, `2 (Duck)` 。

* **C-03 程式語言與框架**
  * 必須使用 **Python 3.8+**。
  * 深度學習框架限定使用 **PyTorch**。
  * 遊戲引擎限定使用 **Pygame**。
---

## 4. 驗收標準 (Acceptance Criteria)

專案完成後，需通過以下標準：

* **AC-01 平均存活能力驗證**
  * 載入最終模型 (`checkpoint_dqn.pth`) 執行 `play.py` 連續 10 場。
    * **通過標準**：10 場分數平均超過 200 分。
   
* **AC-02 最佳存活能力驗證**
  * 載入最終模型 (`checkpoint_dqn.pth`) 執行 `play.py` 連續 10 場。
    * **通過標準**：至少有 1 場分數突破 1000 分。

* **AC-03 學習曲線驗證**
  * 檢查輸出的 `training_result.png` 圖表。
    * **通過標準**：
    * **Score 曲線** 呈現整體上升趨勢 (Upward Trend)。

* **AC-04 系統穩定性**
  * 長時間掛機訓練 (Overnight Training) 不會因記憶體溢出 (OOM) 或邏輯錯誤而崩潰。
  * CSV 記錄檔完整無缺漏。


---
<br>

# 二. 系統架構與實作 (System Architecture and Implementation)

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
## 成果展示
#### 運行play.py 執行推論
demo影片:


https://github.com/user-attachments/assets/7d55f5e9-4c49-434e-a283-624a7ee6098a

#### score與reward學習曲線
<img width="1200" height="1000" alt="Code_Generated_Image (2)" src="https://github.com/user-attachments/assets/e9a1e632-f9dd-43e5-a844-3bc0245e6c43" />

---
## 驗收結果

* **AC-01 存活能力驗證**
✅通過
* **AC-02 最佳存活能力驗證**
❌失敗, `最高分680分`
* **AC-03 學習曲線驗證**
✅通過
* **AC-04 系統穩定性**
✅通過

---
<br>

# 三. Deep Q-Network (DQN) 詳細理論

---

本專案實作了 DeepMind 於 2015 年在 Nature 發表的經典論文 **"Human-level control through deep reinforcement learning"**。
以下詳細解析 DQN 的核心數學架構與 Loss Function 推導過程。


### 1. 核心目標：最佳動作價值函數 (Optimal Action-Value Function)

DQN 的目標是讓 Agent 學會一個策略，以最大化未來的累積獎勵。我們透過 **Bellman Optimality Equation** 來定義「最佳動作價值」：

$$
Q^{\ast}(s, a) = \mathbb{E}_{s^{\prime}} \left[ r + \gamma \max_{a^{\prime}} Q^{\ast}(s^{\prime}, a^{\prime}) \mid s, a \right]
$$

**符號定義：**
* $Q^{\ast}(s, a)$: 在狀態 $s$ 採取動作 $a$，並隨後都採取最佳策略所能獲得的最大預期回報 (Optimal Q-value)。
* $\mathbb{E}_{s^{\prime}}$: 對下一狀態分佈的期望值 (Expectation)。
* $r$: 執行動作後獲得的**立即獎勵 (Immediate Reward)**。
* $\gamma$: **折扣因子 (Discount Factor)**，介於 $[0, 1]$ 之間。決定了 Agent 對未來獎勵的重視程度（0 代表只看當下，1 代表長遠規劃）。
* $s^{\prime}$: 執行動作後轉移到的**下一狀態 (Next State)**。
* $a^{\prime}$: 在下一狀態 $s^{\prime}$ 可能採取的動作。

---

### 2. 函數近似 (Function Approximation)

由於 PAIA 遊戲的狀態空間過大，無法用表格記錄每個 $Q$ 值。但 PAIA 可以直接讀取物件參數，因此我們的 DQN 使用一個 **多層感知器 (Multi-Layer Perceptron, MLP)** 來逼近上述的 $Q^*$ 函數：

$$
Q(s, a; \theta) \approx Q^*(s, a)
$$

**符號定義：**
* $Q(s, a; \theta)$: 我們訓練的 **Q-Network**。
* $\theta$: 神經網路當前的**權重參數 (Weights)**。
* $s$: 輸入狀態向量 $\phi$。在本實作中， $\phi$ 為一個正規化後的 **6 維向量**：

$$
\phi = [\text{Speed}, \text{Dist1}, \text{Dist2}, \text{Crow}, \text{Ducking}, \text{In air}]
$$

  
---

### 3. Loss Function 詳細推導

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

### 4. 梯度更新 (Gradient Descent)

為了最小化 Loss，我們對參數 $\theta$ 進行微分，得到梯度方向並進行更新：

$$
\nabla_{\theta_i} L_i(\theta_i) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} \hat{Q}(\phi', a'; \theta_i^-) - Q(\phi, a; \theta_i) \right) \cdot \nabla_{\theta_i} Q(\phi, a; \theta_i) \right]
$$

**符號意義：**
* 這一項描述了「目標」與「預測」的差距（Error Term）。
* $\nabla_{\theta_i} Q(\dots)$: 這是 Q 網路對於權重的梯度。 
* Optimizer: 雖然原始論文使用 RMSProp，但本專案使用 Adam 優化器 (lr=0.0005) 來執行此更新，以獲得更快的收斂速度。
  
### 5. 參考文獻

1. **Original Paper:** Mnih, V., Kavukcuoglu, K., Silver, D., et al. "Human-level control through deep reinforcement learning." *Nature* 518, 529–533 (2015).




