#  DQN Atari Agent — Project Specification & System Design  
*Inspired by “Playing Atari with Deep Reinforcement Learning (2013)” and  
“Human-level Control through Deep Reinforcement Learning (2015)”*

---

# 1. Requirements（需求）

## 1.1 Functional Requirements（功能需求）
1. 系統須能與 PAIA pingpong 遊戲環境互動。
2. 系統須能以 **遊戲中數據(球位置、球速度等等...)** 作為輸入。
3. 系統須具備以下核心功能：
   -  **動作選擇 (Action Selection)**：基於 DQN 輸出 Q-value 選擇動作。
   -  **強化學習訓練**：包含 Experience Replay、Target Network 等。
   -  **策略評估**：能保存訓練曲線。
4. 系統須支援兩種模式：
   - 訓練模式：與環境互動並學習策略。
   - 推論模式：載入模型後自動遊玩。

---

## 1.2 Performance Requirements（效能需求）
1. 訓練過程至少能在 pingpong 上達到「行為明顯改善」之效果。
2. 回合平均分數需隨訓練步數呈現上升趨勢（參考 Nature 2015 的呈現方式）。
3. 單步推論時間需 < 20 ms，避免遊戲卡頓。

---

## 1.3 Interface Requirements（介面）

### Internal Interface（內部模組介面）
| 模組 | 功能 | 輸入 | 輸出 |
|------|------|------|-------|
| `MLPlay update` | 與PAIA pinpong 遊戲互動 | PAIA pinpong 遊戲內資訊(state) | 擊球板動作(action) |
| `DQN Network` | MLP Q-value 網路 | state | Q(s, a) |
| `ReplayBuffer` | 經驗儲存/取樣 | transition | sampled batch |
| `Trainer` | 更新 Q-network、同步 target | batch | loss, updated weights |
| `Logger` | 記錄分數/影片/曲線 | events | log files |


## 1.4 Constraints（限制）

訓練時間可能受限於 GPU/CPU 資源（可能需要 10+ 小時）。  
Frame-rate 需維持穩定（DQN 對環境延遲敏感）。

## 1.5 Acceptance Criteria（驗收標準）

- 系統能成功與PAIA pingpong 遊戲互動並參與線上對戰。
- 平均回合分數明顯高於隨機策略。
- 推論能讓 Agent 自動遊玩至少一整回合(15分)。

---

# Deep Q-Network (DQN) Implementation

本專案實作了 DeepMind 於 2015 年在 Nature 發表的經典論文 **"Human-level control through deep reinforcement learning"**。
以下詳細解析 DQN 的核心數學架構與 Loss Function 推導過程。


## 1. 核心目標：最佳動作價值函數 (Optimal Action-Value Function)

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

## 2. 函數近似 (Function Approximation)

由於 PAIA 遊戲的狀態空間過大，無法用表格記錄每個 $Q$ 值。但 PAIA 可以直接讀取物件參數，因此我們的 DQN 使用一個 **多層感知器 (Multi-Layer Perceptron, MLP)** 來逼近上述的 $Q^*$ 函數：

$$
Q(s, a; \theta) \approx Q^*(s, a)
$$

**網路架構 (基於 `ml_play.py`):**
* **輸入層**: 8 維特徵向量 (球座標、速度、板子座標、障礙物座標)。
* **隱藏層**: 
    * FC1: 512 neurons (ReLU)
    * FC2: 256 neurons (ReLU)
    * FC3: 128 neurons (ReLU)
* **輸出層**: 動作維度 (Output Size)，對應每個動作的 Q 值。

**符號定義：**
* $Q(s, a; \theta)$: 我們訓練的 **Q-Network**。
* $\theta$: 神經網路當前的**權重參數 (Weights)**。
* $s$: 輸入狀態向量 $\phi$。在本實作中， $\phi$ 為一個正規化後的 **8 維向量**：

$$
phi = [\text{Ball}_x, \text{Ball}_y, \text{Ball}_{vx}, \text{Ball}_{vy}, \text{Paddle}_x, \text{Opponent}_x, \text{Blocker}_x, \text{Blocker}_y]
$$

  ### Q-network 架構圖
  <img width="2856" height="3564" alt="image" src="https://github.com/user-attachments/assets/c4488ed9-c48a-42d0-8772-18b186737431" />
  
---

## 3. Loss Function 詳細推導

為了訓練網路，我們需要定義一個損失函數 (Loss Function) 來衡量預測誤差。DQN 結合了 **Q-Learning** 與 **深度神經網路**，並引入了 **Target Network** 與 **Experience Replay** 來穩定訓練。

### 步驟 3.1: 計算目標值 (The Target)

首先，我們計算對於某個經驗樣本的「標籤」或「目標值」。為了避免網路自我震盪，DQN 使用一個獨立的 **Target Network** 來計算未來的最大價值。

對於第 $j$ 個經驗樣本，目標值 $y_j$ 計算如下：

$$
y_j = r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-)
$$

**符號定義：**
* $y_j$: 第 $j$ 次更新時使用的**目標值 (Target Value)**。
* $r_j$: 該步實際獲得的獎勵。
* $\phi_{j+1}$: 下一步的狀態表示 (預處理後的連續 4 幀圖像)。
* $\hat{Q}$: **目標網路 (Target Network)**，結構與主網路相同，但參數不同。
* $\theta^-$: 目標網路的參數。它是每隔 $C$ 步從主網路參數 $\theta$ 複製過來的（凍結參數），用於提供穩定的訓練目標。

### 步驟 3.2: 損失函數 (The Loss Function)

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

## 4. 梯度更新 (Gradient Descent)

為了最小化 Loss，我們對參數 $\theta$ 進行微分，得到梯度方向並進行更新：

$$
\nabla_{\theta_i} L_i(\theta_i) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} \hat{Q}(\phi', a'; \theta_i^-) - Q(\phi, a; \theta_i) \right) \cdot \nabla_{\theta_i} Q(\phi, a; \theta_i) \right]
$$

**符號意義：**
* 這一項描述了「目標」與「預測」的差距（Error Term）。
* $\nabla_{\theta_i} Q(\dots)$: 這是 Q 網路對於權重的梯度。 
* Optimizer: 雖然原始論文使用 RMSProp，但本專案使用 Adam 優化器 (lr=0.0005) 來執行此更新，以獲得更快的收斂速度。
  
## 5. 參考文獻

1. **Original Paper:** Mnih, V., Kavukcuoglu, K., Silver, D., et al. "Human-level control through deep reinforcement learning." *Nature* 518, 529–533 (2015).

