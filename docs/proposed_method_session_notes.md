# 提案方法：Human-Likeness Style Transfer for OpenVLA-OFT

## 完整架構概要

**目標**：在只用原始 robot demo 微調 OpenVLA-OFT 的情況下，推理時可透過切換 style 條件 S 輸出 human-likeness 風格的機器人動作。

**核心思路**：Content-Style Disentanglement → Frozen Decoder → Latent Distillation

```
Phase 1: 離線訓練 Style-conditioned Deterministic Sequence Autoencoder
  資料: LIBERO-90（paired robot + human demo）
  產出: 凍結的 Encoder + Decoder（神經風格翻譯機）

Phase 2: OpenVLA-OFT 微調 + 雙軌訓練
  資料: LIBERO-Goal（只有 robot demo，無 human demo）
  方法: Frozen Decoder 即時生成 pseudo-human GT；雙軌訓練 Bridge MLP

Phase 3: 零樣本風格切換推理
  輸入: 圖像 + 語言指令 + style S
  輸出: 對應風格的 8 步動作 chunk
```

---

## 背景：OpenVLA-OFT 原始架構

### 模型結構

```
攝影機圖片 → DINOv2 + SigLIP ViT（視覺骨幹）→ patch embeddings（729個）
語言指令 → Llama-2 Tokenizer → token embeddings
proprio → ProprioProjector（2層MLP）→ 1 個嵌入 token

完整序列（拼接）:
[視覺patch_1]...[視覺patch_729] [proprio_token] [文字token_1]... [動作token_1]...

→ Llama-2 Transformer（LoRA 微調）
→ last_hidden_states (B, seq_len, 4096)
→ 用 mask 取動作 token 的 hidden states
→ actions_hidden_states (B, chunk_len × action_dim, 4096)
→ L1RegressionActionHead（MLPResNet：2層殘差MLP）
→ predicted_actions (B, chunk_len, action_dim)
```

### 關鍵程式碼位置

| 元件 | 檔案 | 行數 |
|------|------|------|
| FinetuneConfig | `vla-scripts/finetune.py` | L69-116 |
| VLA forward pass | `prismatic/extern/hf/modeling_prismatic.py` | L499-680 |
| Proprio 處理 | `prismatic/extern/hf/modeling_prismatic.py` | L450-459 |
| Action mask 邏輯 | `prismatic/training/train_utils.py` | L8-40 |
| L1RegressionActionHead | `prismatic/models/action_heads.py` | L84-107 |
| MLPResNet | `prismatic/models/action_heads.py` | L58-81 |
| ProprioProjector | `prismatic/models/projectors.py` | L6-28 |
| FiLMedVisionTransformerBlock | `prismatic/models/film_vit_wrapper.py` | L11-80 |
| Constants | `prismatic/vla/constants.py` | 全檔 |

### 重要常數（LIBERO 設定）

```python
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,     # 動作 chunk 長度（已從 25 改為 8）
    "ACTION_DIM": 8,            # 關節位置(7) + 夾爪(1)
    "PROPRIO_DIM": 8,           # 關節位置(7) + 夾爪寬度(1)
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}
```

### hidden state 提取流程（finetune.py L347-358）

```python
# ① 取 Transformer 最後一層隱藏狀態
last_hidden_states = output.hidden_states[-1]           # (B, seq_len, 4096)

# ② 跳過視覺 patch，只保留文字+動作部分
text_hidden_states = last_hidden_states[:, num_patches:-1]

# ③ 用 mask 只取動作 token 的隱藏狀態
actions_hidden_states = (
    text_hidden_states[current_action_mask | next_actions_mask]
    .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)  # (B, 8×8, 4096) = (B, 64, 4096)
    .to(torch.bfloat16)
)
```

### L1RegressionActionHead 計算流程

```python
# reshape: (B, 64, 4096) → (B, 8, 8×4096) = (B, 8, 32768)
rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)

# MLPResNet: LayerNorm(32768) → Linear(32768→4096) → ReLU
#   → ResBlock(4096) × 2  （每個: LayerNorm→Linear→ReLU + 殘差連接）
#   → LayerNorm(4096) → Linear(4096→8)
action = self.model(rearranged)  # (B, 8, 8)
```

---

## Phase 1：離線鍛造「神經風格翻譯機」

### 目標

將 IK 演算法產生的帶抖動 human-likeness 軌跡，提煉成平滑、精準且受控的神經網路解碼器。獨立離線訓練，不需要 Llama-2 與影像。

### 資料

- **LIBERO-90** 的原始 demo（robot style）
- 用 IK 轉換產生的 human-likeness demo（human style）
- 兩者 **paired**：同任務同 seed，TCP 軌跡（幾乎）相同，僅肘部姿態不同
- TCP rotation 可能有微小差異

### 架構選擇：Deterministic Sequence Autoencoder（非 CVAE）

**理由**：
- LIBERO 同任務 demo 幾乎相同 → 不需要 stochastic z 捕捉多模態性
- Human style 是 IK 確定性算出 → 同 c 同 s 只有一種解
- 避免 KL 的 posterior collapse 風險和額外超參調整
- 未來若需多模態取樣，可加入 `use_vae=True` flag 升級

### 網路架構

```
Encoder（1D-CNN 或 Bi-directional GRU）:
  輸入: X ∈ R^(8×8)（8 步 × 8 維動作）
  輸出: c ∈ R^(8×D_c)（8 個任務碼，D_c 建議 32~64）

Decoder（1D-CNN + FiLM conditioning on s）:
  輸入: c ∈ R^(8×D_c) + s ∈ {"robot", "human"}
  輸出: X̂ ∈ R^(8×8)（8 步 × 8 維動作）

Style s 表示: Learned Embedding（style_dim=32，可學習）
  s_vec = nn.Embedding(num_styles=2, embedding_dim=32)(s)
```

### Loss 函數

$$\mathcal{L}_{Phase1} = \mathcal{L}_{Recon} + \lambda_{FK}\mathcal{L}_{FK} + \beta\mathcal{L}_{MJE} + \gamma\mathcal{L}_{Content} + \eta\mathcal{L}_{Cross}$$

| Loss | 公式 | 作用 |
|------|------|------|
| $\mathcal{L}_{Recon}$ | $\|\hat{X} - X\|_1$ 或 Huber | 基本重建 |
| $\mathcal{L}_{FK}$ | $\text{MSE}(\text{FK}_{pos}(\hat{X}), \text{TCP}^{GT}_{pos})$ | TCP 鎖定，確保夾爪位置不偏 |
| $\mathcal{L}_{MJE}$ | $\sum_t \|\hat{X}_t - 3\hat{X}_{t-1} + 3\hat{X}_{t-2} - \hat{X}_{t-3}\|^2$ | 最小 jerk 約束，壓制高頻抖動 |
| $\mathcal{L}_{Content}$ | $\|c_{reg} - c_{hum}\|^2$ | 強迫 encoder 忽略 style，只編碼任務意圖 |
| $\mathcal{L}_{Cross}$ | $\|\text{Dec}(c_{reg}, s_{hum}) - X_{hum}\|_1 + \|\text{Dec}(c_{hum}, s_{reg}) - X_{reg}\|_1$ | 強化 style 解耦：用同一個 c 切換 s 重建對方 |

**注意事項**：
- $\mathcal{L}_{MJE}$ 只對 7D joint 算 jerk，gripper 可不算或用較弱正則
- $\mathcal{L}_{Content}$ 假設 paired demo 時間步對齊；若有偏移可用 soft-DTW
- $\mathcal{L}_{FK}$ 需要可微分的 FK 函數
- chunk=8 對 MJE 足夠（需至少 4 個時間步：t, t-1, t-2, t-3）

### Phase 1 產出物

- **凍結 Encoder**：`Enc(X) → c`
- **凍結 Decoder**：`Dec(c, s) → X̂`
- 兩者在 Phase 2 中 `requires_grad=False` 但**不是** `torch.no_grad()`（需讓梯度穿過 decoder 回傳到 Bridge）

### Phase 1 驗證（Phase 2 之前必做）

```python
# 在 LIBERO-Goal 上做 sanity check（跨 task suite 泛化能力）
for chunk in LIBERO_Goal_chunks:
    c = Encoder(chunk)
    recon_robot = Decoder(c, s="robot")
    recon_human = Decoder(c, s="human")
    
    print(f"Robot recon error: {L1(recon_robot, chunk)}")    # 應該很小
    print(f"TCP consistency: {L1(FK(recon_human), FK(chunk))}")  # 應該很小
    print(f"Human jerk: {compute_jerk(recon_human)}")         # 應該平滑
```

泛化合理性：style transfer 核心是肘部冗餘自由度的 IK 重映射，屬於運動學操作，與任務語義無關。LIBERO-90 和 LIBERO-Goal 共享同一機器人（7DoF + gripper），動作空間（R^8）相同。

---

## Phase 2：OpenVLA-OFT 微調 + 雙軌訓練

### 目標

將 OpenVLA-OFT / Llama-2 的視覺語言理解能力與 Phase 1 的凍結 Decoder 對接。教 Llama-2 輸出 Decoder 聽得懂的任務碼 ĉ。

### 資料

- **LIBERO-Goal**（只有 robot demo，**沒有 human demo**）
- Human pseudo-GT 在訓練時由凍結 Decoder **即時生成**（on-the-fly）

### 架構改造（相對原始 OpenVLA-OFT）

| 元件 | 原始 | 改造後 |
|------|------|--------|
| ViT 視覺骨幹 | 凍結 | 保持凍結 |
| Llama-2 | LoRA 微調 | 保持 LoRA 微調 |
| ProprioProjector | 2層MLP | → **FiLM-Conditioned ProprioProjector**（加 style 調制） |
| L1RegressionActionHead | MLPResNet → action | → **Bridge MLP → 任務碼 c_hat** |
| — | — | ＋**凍結 Decoder**（Phase 1 產出） |
| — | — | ＋**凍結 Encoder**（Phase 1 產出，當 teacher） |
| — | — | ＋**Style Classifier**（aux loss，新模組） |

### 新增模組：FiLM-Conditioned ProprioProjector

```python
class FiLMedProprioProjector(nn.Module):
    def __init__(self, llm_dim, proprio_dim, style_dim=32):
        super().__init__()
        # 原始 projector 結構（可從舊 checkpoint 載入 fc1, fc2, act_fn1）
        self.fc1 = nn.Linear(proprio_dim, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.act_fn1 = nn.GELU()
        # 新增 FiLM 層（從頭訓練）
        self.film_scale = nn.Linear(style_dim, llm_dim)
        self.film_shift = nn.Linear(style_dim, llm_dim)

    def forward(self, proprio, style_embedding):
        x = self.act_fn1(self.fc1(proprio))
        x = self.fc2(x)
        gamma = self.film_scale(style_embedding)
        beta = self.film_shift(style_embedding)
        x = x * (1 + gamma) + beta   # FiLM: (1+γ)·x + β
        return x
```

**Proprio token 在 VLA 中的位置**：附加在視覺 patch embeddings 後面，作為額外的「感知 token」。在 `modeling_prismatic.py` L450-459 的 `_process_proprio_features` 中，proprio 被投影成 1 個 token → `torch.cat` 到視覺 patch 後面。FiLM 是在投影**之前**就用 style S 調制它。

### 新增模組：Bridge MLP（取代原 L1RegressionActionHead）

```python
class BridgeMLP(nn.Module):
    """將 Llama-2 hidden states 映射為 Phase 1 的任務碼空間"""
    def __init__(self, input_dim=4096, hidden_dim=4096, latent_dim=64, action_dim=8):
        super().__init__()
        # 結構類似原始 MLPResNet，但輸出維度改為 D_c
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * action_dim,  # 4096 × 8 = 32768
            hidden_dim=hidden_dim,              # 4096
            output_dim=latent_dim               # D_c（建議 32~64）
        )

    def forward(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        c_hat = self.model(rearranged)  # (B, 8, D_c)
        return c_hat
```

**D_c 選擇**：建議 32~64。太小表達力不足，太大 latent matching 困難。

### 新增模組：Style Classifier

```python
class StyleClassifier(nn.Module):
    """從 actions_hidden_states 預測 style 標籤"""
    def __init__(self, hidden_dim=4096, num_styles=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),  # 4096 → 1024
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_styles),   # 1024 → 2
        )

    def forward(self, actions_hidden_states):
        # actions_hidden_states: (B, chunk_len × action_dim, hidden_dim)
        pooled = actions_hidden_states.mean(dim=1)   # (B, hidden_dim)
        style_logits = self.classifier(pooled)        # (B, num_styles)
        return style_logits
```

**放在 actions_hidden_states 上（不 detach）**，讓梯度流回 LLM / proprio_projector。用較小的 λ_aux（0.05-0.1）避免 aux loss 主導訓練。

### Style S 的表示方式

```python
style_embedding = nn.Embedding(num_styles=2, embedding_dim=32)
s_vec = style_embedding(s)  # (B, 32)
```

使用 Learned Embedding，style_dim=32，可學習。

### 雙軌訓練：完整時間線

#### Track A（Robot Style）

```
Step 1: 從 LIBERO-Goal dataset 取 training sample
  images_t      = agentview + eye_in_hand
  proprio_t     = X_reg[t]（robot proprio，8維）
  instruction   = "pick up the cup"
  s             = 0（robot）
  X_GT_robot    = X_reg[t : t+8]（未來 8 步 GT robot 動作）

Step 2: FiLM proprio
  proprio_t → FiLM(proprio_t, s=0) → state_token

Step 3: VLA forward
  [ViT patches] [state_token] [text tokens] [8×8 blank action tokens]
  → Llama-2 + LoRA → h_LLM

Step 4: Style aux loss
  actions_hidden_states → Style Classifier → style_logits
  L_aux_A = CE(style_logits, s=0)

Step 5: Bridge → ĉ
  actions_hidden_states → Bridge MLP → ĉ (8, D_c)

Step 6: Latent loss
  c_gold = Frozen_Encoder(X_GT_robot)
  L_latent_A = MSE(ĉ, c_gold)

Step 7: Decode + Action loss
  X̂_robot = Frozen_Decoder(ĉ, s="robot")
  L_action_A = L1(X̂_robot, X_GT_robot)

Step 8: Track A loss
  L_A = L_action_A + λ · L_latent_A
```

#### Track B（Human Style — On-the-fly 生成 pseudo GT）

```
Step 1: 取同一個 training sample（或同 batch 不同 sample）
  images_t      = 同一步的 robot demo 圖像（共用，因 TCP 幾乎一致）
  instruction   = 同一個語言指令
  X_GT_robot    = X_reg[t : t+8]
  s             = 1（human）

Step 2: ★ On-the-fly 生成 pseudo-human 資料
  with torch.no_grad():
      c_gold = Frozen_Encoder(X_GT_robot)
      X_pseudo_hum = Frozen_Decoder(c_gold, s="human")  # (8, 8)
  proprio_t = X_pseudo_hum[0, :]  ← pseudo-human 第一步當 proprio

Step 3: FiLM proprio（human mode）
  proprio_t → FiLM(proprio_t, s=1) → state_token

Step 4: VLA forward
  [ViT patches] [state_token(human)] [text tokens] [8×8 blank action tokens]
  → Llama-2 + LoRA → h_LLM

Step 5: Style aux loss
  actions_hidden_states → Style Classifier → style_logits
  L_aux_B = CE(style_logits, s=1)

Step 6: Bridge → ĉ
  actions_hidden_states → Bridge MLP → ĉ (8, D_c)

Step 7: Latent loss（★ c_gold 和 Track A 一樣！用 robot demo 算）
  c_gold = Frozen_Encoder(X_GT_robot)
  L_latent_B = MSE(ĉ, c_gold)

Step 8: Decode + Action loss
  X̂_human = Frozen_Decoder(ĉ, s="human")
  L_action_B = L1(X̂_human, X_pseudo_hum)  ← GT 是 decoder 自己生成的

Step 9: Track B loss
  L_B = L_action_B + λ · L_latent_B
```

#### 合併 Loss

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_A + (1 - \alpha) \cdot \mathcal{L}_B + \lambda_{aux} \cdot \mathcal{L}_{aux}$$

- **α = 0.8**（robot GT 主導訓練，品質可靠）
- **(1-α) = 0.2**（pseudo-human GT 輔助，教 FiLM + Classifier 學 style）
- **λ_aux = 0.05~0.1**（輕微暗示，不過度干擾主任務）
- 等價做法：用 batch 比例控制（80% robot / 20% human），loss 不加權

### 圖像共用的合理性

| 攝影機 | Robot vs Human 差異 | 能否共用 |
|--------|---------------------|----------|
| agentview（第三人稱） | 幾乎相同：TCP 位置一致、物體位置一致、唯一差異是肘部高低（很小） | ✅ 直接共用 |
| eye_in_hand（腕部） | TCP position 相同，TCP rotation 有微小差異 → 畫面可能稍微旋轉 | ✅ 可共用（微小差異反而增加魯棒性） |

若 TCP rotation 差異 > ~15°，考慮 Track B 不使用 eye_in_hand。

### L_latent 和 L_action 的權重排程

```
訓練初期: λ_Latent = 1.0,  λ_Action = 0.1  （先學「說 decoder 的語言」）
訓練中期: λ_Latent = 0.5,  λ_Action = 0.5  （平衡過渡）
訓練後期: λ_Latent = 0.1,  λ_Action = 1.0  （最終行為品質主導）
```

### Scheduled Sampling（解決 Exposure Bias）

訓練後期逐步引入 rollout proprio：

```
訓練初期（teacher forcing）:
  Track B: proprio_t = X_pseudo_hum[0, :]（frozen decoder 生成的）

訓練後期（scheduled sampling，機率 p 逐步下降）:
  以機率 p:   餵 X_pseudo_hum[0, :]（穩定）
  以機率 1-p: 餵 模型自己 decode 出的第一步（與部署一致）
    → ĉ = Bridge(h_LLM)
    → X̂_hum = Frozen_Decoder(ĉ, s="human")
    → proprio_{t+1} = X̂_hum[0, :]

p 排程: 1.0 → 0.2 → 0.0
```

### Decoder 凍結的梯度注意事項

```python
# 正確做法：requires_grad=False，梯度仍穿過 decoder 流向 Bridge
for param in frozen_decoder.parameters():
    param.requires_grad = False

# 錯誤做法：torch.no_grad() 會切斷梯度流
# with torch.no_grad():
#     x_hat = decoder(c_hat, s)  ← 不要這樣做！
```

### Phase 2 不需要的 Loss

- **不需要 L_MJE**（平滑已固化在凍結 decoder）
- **不需要 L_FK**（運動學鎖定已固化在凍結 decoder）

---

## Phase 3：零樣本風格切換推理

### 執行流程

```
1. 使用者給指令 "Pick up the cup"，設定 s="human"
2. 當下關節角 proprio_t 經 FiLM(proprio_t, s=1) → state_token
3. [ViT patches] [state_token] [text tokens] [8×8 blank action tokens]
   → Llama-2 + LoRA → h_LLM
4. Bridge MLP → ĉ ∈ R^(8×D_c)
5. Frozen Decoder(ĉ, s="human") → 8 步 human-style 動作
6. 執行第 1 步 → FK 計算新 proprio → 回到步驟 2
```

### 技術優勢

- **無延遲風格轉換**：IK、平滑、運動學約束已固化在權重中，推理只需矩陣運算
- **chunk=8 平行解碼**：一次 VLA forward 產生 8 步動作

---

## Checkpoint 相容性分析

### Phase 2 可複用的舊 checkpoint

| 元件 | 能否複用？ | 方法 |
|------|-----------|------|
| **LoRA 適配器** | ✅ 直接載入 | Transformer 結構不變 |
| **ProprioProjector** | ⚠️ 部分載入 | `strict=False`：fc1, fc2 保留，FiLM 層新訓 |
| **L1RegressionActionHead** | ⚠️ 包進新 class | `{f"base.{k}": v}` 前綴映射 |
| **CVAE/AE encoder/decoder** | ✅ 新模組 | Phase 1 產出，直接凍結使用 |
| **Style Classifier** | ✅ 新模組 | 從頭訓練 |
| **Bridge MLP** | ✅ 新模組 | 從頭訓練 |

### Checkpoint 載入範例

```python
# ProprioProjector：部分載入
old_state_dict = load_checkpoint("proprio_projector", path, step)
new_proprio_projector.load_state_dict(old_state_dict, strict=False)

# Action Head → Bridge MLP：如果結構相同可用前綴映射
old_state_dict = load_checkpoint("action_head", path, step)
new_state_dict = {f"base.{k}": v for k, v in old_state_dict.items()}
bridge_mlp.load_state_dict(new_state_dict, strict=False)
```

### 改 NUM_ACTIONS_CHUNK（25→8）的影響

- action_head 的 MLPResNet input_dim = 4096 × ACTION_DIM = 32768 **不受影響**（per-timestep）
- LoRA 權重不受影響（attention weight matrix 大小與序列長度無關）
- Transformer 序列長度不同（25×8=200 → 8×8=64 個動作 token），但不影響權重

---

## 關鍵設計決策摘要

| 決策 | 選擇 | 理由 |
|------|------|------|
| AE 類型 | Deterministic（非 CVAE+KL） | LIBERO demo 幾乎一致，不需 stochastic z；避免 posterior collapse |
| Phase 1 資料 | LIBERO-90 paired | Phase 2 用不同 task suite (LIBERO-Goal) 測試泛化 |
| Phase 2 human GT | On-the-fly 由 frozen decoder 即時生成 | 不需預先做 IK 轉換；計算成本 <1ms |
| c_gold 來源 | 兩軌都用 robot demo 算 | 保持一致性；content loss 保證 c_reg ≈ c_hum |
| 圖像 | Track A/B 共用 robot 圖像 | TCP 位置一致，視角幾乎相同 |
| Style 表示 | Learned Embedding (dim=32) | 可學習、未來易擴展 |
| Decoder 凍結 | requires_grad=False（非 no_grad） | 讓梯度穿過 decoder 流向 Bridge |
| Track A/B 權重 | α=0.8 / (1-α)=0.2 | Robot GT 可靠主導；pseudo-human GT 輔助學 style |
| D_c | 32~64 | 平衡表達力與 latent matching 難度 |

---

## 風險與緩解

| 風險 | 嚴重度 | 緩解方法 |
|------|--------|----------|
| Phase 1 AE 在 LIBERO-Goal 泛化失敗 | 🔴 高 | Phase 2 前做 sanity check；若失敗則 Phase 1 加入更多樣動作資料 |
| IK 產生的 human demo 品質差 | 🔴 高 | Phase 1 前可視化 paired demo；確認 FK/jerk/奇異點 |
| Pseudo-human GT 的 noise 被 VLA 放大 | 🟡 中 | α=0.8 讓 robot GT 主導；L_latent 比 L_action 更重要在 Track B |
| Exposure bias（rollout proprio 分佈偏移） | 🟡 中 | Scheduled Sampling / DAgger-style state mixing |
| Style Classifier 過度干擾主任務 | 🟢 低 | λ_aux=0.05-0.1 控制 |

---

## 建議的實驗順序

```
1. 準備 LIBERO-90 的 paired data（robot + IK human）
2. 可視化 paired demo，確認品質

3. 訓練 Phase 1 AE
4. 驗證 Phase 1：cross-style reconstruction 品質
5. 驗證 Phase 1：在 LIBERO-Goal 上的泛化能力（sanity check）

6. Phase 2：先只用 L_latent 訓練 Bridge，確認 ĉ ≈ c_gold
7. Phase 2：加入 L_action 和 dual-track
8. Phase 2：加入 Style Classifier aux loss
9. Phase 2：加入 Scheduled Sampling

10. Phase 3：測試零樣本風格切換
```

---

## 附錄：完整架構圖

```
                              Style S ∈ {robot, human}
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
    ┌──────────────────┐  ┌───────────────┐   ┌──────────────────┐
    │ FiLM-Conditioned │  │ Frozen        │   │ Style Classifier │
    │ ProprioProjector │  │ Decoder       │   │ (aux loss)       │
    │ (修改 proprio)   │  │ (Phase 1 產出)│   │ (新模組)         │
    └────────┬─────────┘  └───────┬───────┘   └────────┬─────────┘
             │                    ↑                     │
    附加到視覺patch嵌入     接收 ĉ + s              從 h_LLM 分類 s
             │                    │                     │
             ▼                    │                     │
    ┌─────────────────────────────────────────────────────────────┐
    │ OpenVLA-OFT                                                 │
    │ [ViT patches] [state_token] [text tokens] [action tokens]  │
    │ → Llama-2 + LoRA → h_LLM → actions_hidden_states          │
    └──────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
              ┌──────────────┐      ┌──────────────────┐
              │ Bridge MLP   │      │ Frozen Encoder   │
              │ h → ĉ       │      │ X_GT → c_gold    │
              └──────┬───────┘      └────────┬─────────┘
                     │                       │
                     └───── L_latent ────────┘
                     │
                     ▼
              Frozen Decoder(ĉ, s)
                     │
                     ▼
              X̂ (predicted actions)
                     │
                     ▼
              L_action = L1(X̂, X_GT or X_pseudo_hum)
```
