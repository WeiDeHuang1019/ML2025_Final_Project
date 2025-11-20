#  DQN Atari Agent — Project Specification & System Design  
*Inspired by “Playing Atari with Deep Reinforcement Learning (2013)” and  
“Human-level Control through Deep Reinforcement Learning (2015)”*

---

# 1. Requirements（需求）

## 1.1 Functional Requirements（功能需求）
1. 系統須能與 Atari 2600 遊戲環境互動（Gymnasium / ALE）。
2. 系統須能以 **畫面像素**作為輸入，不使用遊戲內部狀態。
3. 系統須具備以下核心功能：
   -  **畫面擷取與前處理**：灰階化、縮放至 84×84、frame-stacking。
   -  **動作選擇 (Action Selection)**：基於 DQN 輸出 Q-value 選擇動作。
   -  **強化學習訓練**：包含 Experience Replay、Target Network 等。
   -  **策略評估與影片紀錄**：能保存訓練曲線與玩遊戲回放影片。
4. 系統須支援兩種模式：
   - 訓練模式：與環境互動並學習策略。
   - 推論模式：載入模型後自動遊玩。

---

## 1.2 Performance Requirements（效能需求）
1. 訓練過程至少能在 Pong 或 Breakout 上達到「行為明顯改善」之效果。
2. 回合平均分數需隨訓練步數呈現上升趨勢（參考 Nature 2015 的呈現方式）。
3. 訓練至少需支援 200k ~ 1M 環境 steps。
4. 單步推論時間需 < 20 ms，避免遊戲卡頓。
5. Replay Buffer 最低容量 100,000 條 transition（推薦 1,000,000）。

---

## 1.3 Interface Requirements（介面）

### Internal Interface（內部模組介面）
| 模組 | 功能 | 輸入 | 輸出 |
|------|------|------|-------|
| `Env Wrapper` | 處理 Atari 前處理、frame-stack | raw frame | 84×84×4 state |
| `DQN Network` | CNN Q-value 網路 | state | Q(s, a) |
| `ReplayBuffer` | 經驗儲存/取樣 | transition | sampled batch |
| `Trainer` | 更新 Q-network、同步 target | batch | loss, updated weights |
| `Logger` | 記錄分數/影片/曲線 | events | log files |

### External Interface（外部介面）
- CLI 指令介面，例如：  
  ```bash
  python train.py --env PongNoFrameskip-v4 --steps 500000
  python eval.py --model checkpoint.pth

## 1.4 Constraints（限制）

訓練時間可能受限於 GPU/CPU 資源（無 GPU 可能需要 10+ 小時）。
Frame-rate 需維持穩定（DQN 對環境延遲敏感）。

## 1.5 Acceptance Criteria（驗收標準）

系統能成功與至少一款 Atari 遊戲互動（Pong/Breakout）。
訓練至少 200k steps（建議 500k–1M）。
平均回合分數明顯高於隨機策略。
推論能讓 Agent 自動遊玩至少一整回合。
需提供：
- 學習曲線
- 玩遊戲之影片
- 主要設計文件（FSM、架構圖、程式結構）

