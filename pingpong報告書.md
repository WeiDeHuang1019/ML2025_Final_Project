#  DQN Atari Agent — Project Specification & System Design  
*Inspired by “Human-level Control through Deep Reinforcement Learning (2015)”* <br>
本報告分成三部分 <br>
> 一. 專案規格 <br>
> 二. 系統架構與實作 <br>
> 三. Deep Q-Network (DQN) 詳細理論 <br>

---
<br>

# 一. 專案規格 (Project Specification)

---

##  Functional Requirements（功能需求）
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

##  Performance Requirements（效能需求）
1. 訓練過程至少能在 pingpong 上達到「行為明顯改善」之效果。
2. 回合平均分數需隨訓練步數呈現上升趨勢（參考 Nature 2015 的呈現方式）。
3. 單步推論時間需 < 20 ms，避免遊戲卡頓。

---

##  Constraints（限制）

1. 不能使用任何Rule-Based設計。  
2. 必須使用PAIA平台之遊戲環境以及其提供之遊戲狀態(state)。
3. 神經網路設計不可超過五層(避免冗餘設計)。

---

##  Acceptance Criteria（驗收標準）
### 於PAIA pingpong環境: 
- 系統能成功與PAIA pingpong 遊戲互動並參與線上對戰。
- 參與線上對戰能穩定存活到到速度20
- 能讓 Agent 自動對打遊玩至少達15個回合(一個來回為1回合)。

-----
<br>

# 二. 系統架構與實作 (System Architecture and Implementation)

本專案使用強化學習（Reinforcement Learning）中的 **Deep Q-Network (DQN)** 演算法，訓練一個能夠自動遊玩 PAIA 乒乓球遊戲的 AI Agent，在此部分將說明本專案整體設計的架構與實作細節以及超參數設計。

---
## BreakDown
<img width="4280" height="1284" alt="image" src="https://github.com/user-attachments/assets/10102b59-6244-4a05-8aa9-7cbdbb7c7f6d" />

---
## Architecture
<img width="3888" height="3044" alt="image" src="https://github.com/user-attachments/assets/08df9f76-bf63-462e-b696-ef488eaf4ecd" />

---
## APIs
| 檔案名稱 | 1. 輸入 (Input) | 2. 輸出 (Output) | 3. 主要參數 (Param) | 4. 方法/邏輯 (Method) | 5. 呼叫例子 (Call Example) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **game\_config.json** | 無 (靜態設定檔) | **命令列參數定義**<br>(CLI Arguments Schema) | `difficulty` ("EASY"/"NORMAL"/"HARD")<br>`game_over_score` (1-15)<br>`init_vel` (1-30) | **定義規格**：<br>設定參數名稱、縮寫、型別、預設值與允許範圍。 | `python -m mlgame ... --difficulty HARD` |
| **config.py** | `src.game.PingPong` 類別 | **`GAME_SETUP` 字典** | `GAME_SETUP["game"]` | **註冊入口**：<br>將遊戲的主類別 (`PingPong`) 映射給 MLGame 框架。 | `game_cls = config.GAME_SETUP["game"]` |
| **src/game.py**<br>(遊戲核心) | **指令 (Commands)**<br>`{"1P": "MOVE_LEFT", ...}` | **場景資訊 (Scene Info)**<br>`{"ball": (x,y), ...}`<br>**渲染資料** `scene_progress` | `difficulty`<br>`game_over_score`<br>`init_vel` | **`update()`**: 更新遊戲狀態。<br>**`get_data_from_game_to_player()`**: 打包給 AI 的資料。<br>**`_game_over()`**: 判定勝負。 | `scene_info = game.get_data_from_game_to_player()`<br>`result = game.update(commands)` |
| **src/game\_object.py**<br>(物理實體) | **物理屬性**<br>(碰撞判定、移動方向) | **物件狀態**<br>(座標 `rect`、速度 `speed`) | `init_pos` (初始位置)<br>`play_area_rect` (邊界)<br>`side` ("1P"/"2P") | **`move()`**: 計算位移。<br>**`check_bouncing()`**: 處理球的反彈與切球物理。<br>**`reset()`**: 重置位置。 | `ball.move()`<br>`ball.check_bouncing(platform_1P, ...)` |
| **ml/ml\_play.py**<br>(AI 訓練端) | **場景資訊 (Scene Info)**<br>(球座標、板子位置) | **動作指令 (Command)**<br>`"MOVE_LEFT"`, `"NONE"`<br>**模型檔案** (`dqn_model.pth`) | `batch_size=1024`<br>`lr=0.0005`<br>`epsilon` (探索率) | **`update()`**: 決定動作並呼叫訓練。<br>**`train_step()`**: 計算 Loss 更新網路。<br>**`get_state_vector()`**: 特徵正規化。<br>**`save_model()`**: 儲存權重。 | `command = ai.update(scene_info)`<br>`ai.save_model()` |
| **ml/ml\_play\_eval.py**<br>(AI 推論端) | **場景資訊 (Scene Info)**<br>**模型權重** (`dqn_model.pth`) | **動作指令 (Command)**<br>(僅回傳動作，不產出模型) | `epsilon=0.05`<br>(極低探索率，趨向貪婪策略) | **`_load_model_for_eval()`**: 載入訓練好的權重。<br>**`select_action()`**: 僅使用 `argmax` 選擇最佳動作，不進行反向傳播訓練。 | `ai = MLPlay("1P")`<br>`command = ai.update(scene_info)` |

#### 表格重點補充

1.  **資料流向 (Data Flow)**:
      * **輸入端**: `game_config.json` 定義了參數，由 `config.py` 傳遞給 `game.py` 初始化環境。
      * **迴圈中**: `game.py` 產出 `scene_info` (Output) -\> 傳入 `ml_play.py` (Input) -\> 經過神經網路運算 -\> 產出 `Command` (Output) -\> 傳回 `game.py` (Input) 更新畫面。
2.  **訓練 vs 推論**:
      * `ml_play.py` 的特點是包含 **`ReplayMemory`** 與 **`train_step`**，且會輸出 `.pth` 檔。
      * `ml_play_eval.py` 則移除了訓練邏輯，只包含 **`load_model`** 與 **`select_action`**，專注於使用已存在的 `.pth` 檔進行遊戲。
3.  **物件層級**:
      * `game.py` 是主要遊戲進行，它實例化並管理 `game_object.py` 裡面的 `Ball`, `Platform`, `Blocker`。

---
## 模型架構 (Q-Net Architecture)

#### 網路結構 
採用全連接層 (Fully Connected Layers)，結構為 **深層寬體網路**：
* **Input Layer**: 8 neurons (對應狀態向量)
* **Hidden Layer 1**: 512 neurons (ReLU) - *加寬以提取更多特徵*
* **Hidden Layer 2**: 256 neurons (ReLU)
* **Hidden Layer 3**: 128 neurons (ReLU) - *加深以處理非線性障礙物邏輯*
* **Output Layer**: 3 neurons (對應動作：左移、右移、不動)

#### Q-network 架構圖
<img width="2856" height="3564" alt="image" src="https://github.com/user-attachments/assets/c4488ed9-c48a-42d0-8772-18b186737431" />

#### 訓練更新 (Optimization)
* 使用 **Adam Optimizer** 進行梯度下降。
* 損失函數採用 **MSE Loss** (均方誤差)，計算預測 Q 值與目標 Q 值 ( $R + \gamma \max Q(s', a')$ ) 之間的差異。

####  狀態向量 (State Vector, $\phi$)

為了讓神經網路理解當前的遊戲局勢，我們將遊戲畫面資訊 (`scene_info`) 轉換為一個長度為 **8** 的正規化特徵向量 $\phi(s)$。所有數值皆經過標準化處理（除以螢幕寬高或最大速度），以加速神經網路收斂。

輸入狀態向量定義如下：

| Index | 特徵名稱 | 描述 | 正規化方式 |
| :--- | :--- | :--- | :--- |
| **0** | $Ball_x$ | 球的 X 座標 | $x / 200$ |
| **1** | $Ball_y$ | 球的 Y 座標 | $y / 500$ |
| **2** | $V_x$ | 球的水平速度 | $v_x / 20$ |
| **3** | $V_y$ | 球的垂直速度 | $v_y / 20$ |
| **4** | $Paddle_{Self}$ | 我方板子 X 座標 | $x / 200$ |
| **5** | $Paddle_{Op}$ | 對手板子 X 座標 | $x / 200$ |
| **6** | $Blocker_x$ | 障礙物 X 座標 | $x / 200$ (若無則為負值) |
| **7** | $Blocker_y$ | 障礙物 Y 座標 | $y / 500$ (若無則為負值) |


---
## 遊戲模式適應機制 (Mode Adaptation)

本模型採用 **統一狀態表示法 (Unified State Representation)** 來同時處理「普通模式」與「困難模式」，無需手動切換程式碼。

* **普通模式 (Normal Mode)**：
    當場景中沒有障礙物時，程式會捕捉到異常或空值，此時狀態向量中的 `Blocker` 座標 (Index 6, 7) 會被設為預設值（如 -1）。神經網路會學習到當這兩個數值為負時，只需專注於球與板子的互動。
* **困難模式 (Hard Mode)**：
    當場景中出現障礙物（Blocker）時，其真實座標會被正規化並填入向量。由於我們加深了網路層數（詳見模型架構），DQN 能夠學習障礙物的軌跡特徵，並發展出避開或利用障礙物反彈的策略。


---
## 獎勵機制 (Reward Function)

為了引導 Agent 學習正確的行為，我們設計了一套 **稀疏獎勵 (Sparse Reward)** 與 **塑形獎勵 (Shaping Reward)** 混合的機制 $R$：

#### A. 勝負獎勵 (Outcome Reward)
這是最核心的目標，發生在回合結束時：
* **獲勝 (+20)**：成功將球回擊且對手未接到（球通過對手防線）。
* **落敗 (-20)**：未能接到球（球通過我方防線）。

#### B. 引導獎勵 (Shaping Reward)
為了加速初期訓練，避免 Agent 像無頭蒼蠅一樣隨機移動：
* **追球獎勵 (+0.1)**：當我方板子中心與球的水平距離 (`dist`) 小於 20 pixels 時給予微小獎勵。這鼓勵 Agent 隨時保持在球的下方。
* **有效回擊 (+5)**：當偵測到球的垂直速度 ($V_y$) 方向改變（代表發生碰撞）且球位於特定高度區間時，視為成功回擊。這讓 Agent 學習到「碰到球」是高價值的行為。


---
## 關鍵超參數設計 (Hyperparameters)

為了在效能與訓練穩定度之間取得平衡，主要超參數設定如下：

| 參數 | 數值 | 說明 |
| :--- | :--- | :--- |
| **Batch Size** | `1024` | 配合高階 GPU 的大批次訓練，能讓梯度估計更穩定。 |
| **Learning Rate** | `0.0005` | 學習率。配合大 Batch Size 進行微調。 |
| **Gamma ($\gamma$)** | `0.99` | 折扣因子。重視長期利益（贏得比賽）而非僅是短期回報。 |
| **Memory Capacity** | `100,000` | 經驗回放緩衝區大小，儲存過往的 `(s, a, r, s')`。 |
| **Target Update** | `1000` | 每 1000 步更新一次 Target Network，保持訓練目標穩定。 |

---
## 探索策略 ($\epsilon$-Greedy Strategy)

$\epsilon$ (Epsilon) 控制著 Agent 在「探索 (Exploration)」與「利用 (Exploitation)」之間的平衡。

* **初始值**: `1.0` (100% 隨機動作，完全探索)
* **最小值**: `0.05` (保留 5% 的機率進行隨機嘗試)
* **衰減率 (Decay)**: `0.99992`

**機制說明**：
在訓練初期，$\epsilon$ 很高，Agent 會隨機亂動以收集各種可能的狀態與結果。隨著訓練進行，$\epsilon$ 每次與環境互動後都會乘以 `0.99992` 慢慢下降。這是一個非常緩慢的衰減過程，確保 Agent 在收斂到固定策略之前，有足夠長的時間去探索各種複雜的球路與障礙物反彈情況。

---

### 模型學習過程-折線圖
![S__82149427](https://github.com/user-attachments/assets/24fcfa03-756c-4169-87e4-28a9f13e67e4)

- **Episode Reward (Sum) - 回合總獎勵**：代表每一局遊戲（Episode）結束時，Agent 獲得的總分。
- **Average Q-Value - 平均 Q 值**：這是神經網路對於「採取某個動作後，預期未來能拿多少分」的估計值平均。
- **Total Loss per Episode - 每回合總損失**：代表預測值（Q_predict）與目標值（Q_target）之間的誤差。
- **Weight vs Loss - 權重與損失關係圖**：觀察參數空間的變化與 Loss 的關係。


---
<br>

# 三. Deep Q-Network (DQN) 詳細理論


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
* $s$: 輸入狀態向量 $\phi$。在本實作中， $\phi$ 為一個正規化後的 **8 維向量**：

$$
phi = [\text{Ball}_x, \text{Ball}_y, \text{Ball}_{vx}, \text{Ball}_{vy}, \text{Paddle}_x, \text{Opponent}_x, \text{Blocker}_x, \text{Blocker}_y]
$$

  
---

### 3. Loss Function 詳細推導

為了訓練網路，我們需要定義一個損失函數 (Loss Function) 來衡量預測誤差。DQN 結合了 **Q-Learning** 與 **深度神經網路**，並引入了 **Target Network** 與 **Experience Replay** 來穩定訓練。

#### 步驟 3.1: 計算目標值 (The Target)

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

#### 步驟 3.2: 損失函數 (The Loss Function)

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












