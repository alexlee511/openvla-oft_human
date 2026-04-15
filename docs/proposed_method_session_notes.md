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
- 用 IK 轉換產生的 human-likeness demo（human style），由 `A_preprocess_human_demo_14.py` + `A_elbow_projector_20.py` 生成
- 兩者 **paired**：同任務同 seed，TCP 軌跡（幾乎）相同，僅肘部姿態不同
- TCP rotation 可能有微小差異
- **Gripper commands 完全相同**：humanization 只重映射 null-space joints，不改 gripper
- 資料來源：`humanized_sim.npz`（preferred）或 `humanized.npz`（fallback）
  - `joint_states_demo` (T, 7) — 原始 robot joint positions
  - `joint_states_human` (T, 7) — humanized joint positions（IK commanded，非 sim actual）
  - `gripper_commands` (T,) — gripper commands（兩種 style 共用）
- 使用 `joint_states_human`（commanded）而非 `joint_states_sim`（sim actual），因為 sim 版有控制器 tracking lag（~2° 平均，最大 24°）會引入噪音
- 正規化：Panda joint limits → [-1, 1]，gripper 已在 [-1, 1]
- 驗證集切分：按 **task** 分割（90% tasks → train, 10% → val），確保 val 測試的是對未見任務的泛化

### 架構選擇：Deterministic Sequence Autoencoder（非 CVAE）

**理由**：
- LIBERO 同任務 demo 幾乎相同 → 不需要 stochastic z 捕捉多模態性
- Human style 是 IK 確定性算出 → 同 c 同 s 只有一種解
- 避免 KL 的 posterior collapse 風險和額外超參調整
- 未來若需多模態取樣，可加入 `use_vae=True` flag 升級

### 預設超參數（`configs/default.yaml`）

```yaml
model:
  encoder_type: bigru     # bigru | cnn
  latent_dim: 64          # D_c: 每個 timestep 的 content code 維度
  hidden_dim: 256         # Encoder / Decoder 隱藏維度
  style_dim: 32           # Style embedding 維度
  num_styles: 2           # robot=0, human=1
  encoder_num_layers: 2   # BiGRU 層數 (或 CNN conv blocks)
  encoder_dropout: 0.0
  decoder_num_blocks: 3   # FiLM-conditioned conv blocks
  decoder_kernel_size: 3

loss:
  lambda_recon: 1.0       # 重建（joints only）
  lambda_fk: 5.0          # TCP position lock
  lambda_mje: 0.1         # Minimum jerk（joints only）
  lambda_content: 1.0     # c_robot ≈ c_human
  lambda_cross: 1.0       # Cross-style reconstruction（full 8D）
  lambda_gripper: 2.0     # 顯式 gripper 保存
  lambda_cross_elbow: 0.5 # Cross-branch elbow geometric loss

data:
  chunk_len: 8            # 匹配 OpenVLA-OFT NUM_ACTIONS_CHUNK
  stride: 4               # 滑動窗口步長
  normalize: true         # 用 Panda joint limits 正規化到 [-1, 1]

training:
  epochs: 200
  batch_size: 512
  lr: 1.0e-3
  lr_scheduler: cosine
  warmup_epochs: 5
  grad_clip: 1.0
```

### 網路架構（已實作於 `SCAE/scae/`）

#### 標準 SCAE

```
Encoder（BiGRU 或 1D-CNN，由 config 選擇）:
  輸入: X ∈ R^(8×8)（8 步 × 8 維動作 = 7 joints + 1 gripper）
  輸出: c ∈ R^(8×D_c)（8 個任務碼，D_c = 64）
  ★ Encoder 看到完整 8D（含 gripper 作為上下文），因為 gripper 狀態
    影響 approach_blend，而 approach_blend 直接決定了 human joints 與
    demo joints 的差異程度。讓 encoder 能感知 gripper 上下文有助於
    產生更好的 content code。

Decoder（1D-CNN + FiLM conditioning on s）:
  輸入: c ∈ R^(8×D_c) + s ∈ {"robot", "human"}
  輸出: X̂ ∈ R^(8×8)（8 步 × 8 維動作 = 7 joints + 1 gripper）
  ★ Decoder 輸出完整 8D。FiLM 層初始化為近 identity（γ≈0, β≈0），
    因此 style 對 gripper 的調制天然趨近零；加上 L_Gripper 顯式監督，
    確保 gripper 信號忠實穿透。

Style s 表示: Learned Embedding（style_dim=32，可學習）
  s_vec = nn.Embedding(num_styles=2, embedding_dim=32)(s)
```

#### GeoSCAE（Geometry-Aware Residual 變體，`scae/scae_geo.py`）

與標準 SCAE 共享 Encoder，但 Decoder 改為**殘差 + 幾何條件化**：

```
GeoSCAE 結構:
  Encoder: 完全相同的 BiGRU / CNN → c ∈ R^(8×D_c)

  Geometric Feature Extractor:
    輸入: X_input ∈ R^(8×8)（原始輸入動作 chunk）
    → denormalize → FK(joint_7D) → 4 個 keypoints (shoulder, elbow, wrist, TCP)
    → concatenate: (B, T, 12) → GeoMLP → geo ∈ R^(B, T, geo_dim=32)

  Residual Decoder（FiLMGeoDecoder）:
    輸入: c ∈ R^(8×D_c) + geo ∈ R^(8×geo_dim) + s_vec ∈ R^32
    輸出: Δ ∈ R^(8×8)（殘差預測）
    最終輸出: X̂ = X_input + Δ

  ★ 殘差公式 + 幾何條件化的優點：
    1. 學的是 style delta（Δ）而非絕對 joint 配置 → OOD 泛化更好
    2. FK keypoints 提供 Cartesian 空間錨點 → decoder 知道手臂幾何姿態
    3. 對未見過的 joint 配置，殘差自然趨近零 → 安全降級
```

**GeoSCAE vs SCAE 差異**：
| 特性 | SCAE | GeoSCAE |
|------|------|---------|
| Decoder 類型 | FiLMDecoder（直接輸出 X̂） | FiLMGeoDecoder（預測殘差 Δ） |
| 幾何條件 | 無 | FK keypoints → GeoMLP → geo features |
| 輸出公式 | X̂ = Dec(c, s) | X̂ = X_input + Dec(c, geo, s) |
| Decode 需要 X_input | 否 | **是**（用於 FK 和殘差基準） |
| Config 選擇 | `model.model_type = "default"` | `model.model_type = "geo"` |

**Gripper 設計決策**：
- Robot 與 human demo 的 gripper commands **完全相同**（humanization 只改 joint 姿態）
- Encoder 將 gripper 視為**上下文**（contextual input），而非 style-dependent 信號
- 在 loss 層面，joints 和 gripper **分開監督**：
  - `L_Recon` 只算 joints (dim 0–6)
  - `L_Gripper` 獨立監督 gripper (dim 7)，使用較高權重 λ=2.0
- 此設計讓 arm-style 學習不受 gripper 梯度干擾，同時保證 gripper 精度

### 關鍵程式碼位置（SCAE）

| 元件 | 檔案 | 說明 |
|------|------|------|
| SCAE 主模型 | `scae/scae.py` | Encoder + Decoder + Style Embedding 組裝 |
| GeoSCAE 主模型 | `scae/scae_geo.py` | Encoder + FiLMGeoDecoder + FK + 殘差預測 |
| BiGRU / CNN Encoder | `scae/encoder.py` | 兩種 encoder 實作 + `build_encoder` 工廠函式 |
| FiLM Decoder | `scae/decoder.py` | FiLMLayer → FiLMConv1dBlock → FiLMDecoder |
| GeoMLP + FiLMGeoDecoder | `scae/decoder_geo.py` | 幾何條件化殘差 Decoder |
| Loss 函數 | `scae/losses.py` | 7 項 loss 的計算邏輯 |
| Config | `scae/config.py` | DataConfig / ModelConfig / LossConfig / TrainingConfig |
| 可微分 FK | `scae/panda_fk.py` | Panda 7-DOF DH FK，含 TCP + Elbow + Shoulder + Wrist |
| Dataset | `scae/dataset.py` | Paired action chunk 載入 + 正規化 |
| 訓練腳本 | `train.py` | Phase 1 完整訓練流程 |
| 評估腳本 | `evaluate.py` | Sanity check + 可視化 |
| 預設設定 | `configs/default.yaml` | 所有超參數預設值 |
| LIBERO Pipeline 入口 | `LIBERO/scripts/A_humanized_libero_suite.py` | 整套 suite 的 SCAE humanize 調度 |
| LIBERO-10 並行腳本 | `LIBERO/scripts/A_humanized_libero_10.py` | 平行 SCAE 轉換，per-worker 模型快取 |
| 單 demo Pipeline | `LIBERO/scripts/A_run_pipeline.py` | 單一 demo 的 SCAE 轉換步驟 |

### Loss 函數

$$\mathcal{L}_{Phase1} = \lambda_{r}\mathcal{L}_{Recon} + \lambda_{fk}\mathcal{L}_{FK} + \lambda_{mje}\mathcal{L}_{MJE} + \lambda_{c}\mathcal{L}_{Content} + \lambda_{x}\mathcal{L}_{Cross} + \lambda_{g}\mathcal{L}_{Gripper} + \lambda_{e}\mathcal{L}_{CrossElbow}$$

| Loss | 公式 | 作用 | 預設 λ |
|------|------|------|--------|
| $\mathcal{L}_{Recon}$ | $\|\hat{X}_{joints} - X_{joints}\|_1$（**僅 7D joints**） | 基本重建（不含 gripper，避免梯度干擾） | 1.0 |
| $\mathcal{L}_{FK}$ | $\text{MSE}(\text{FK}_{tcp}(\hat{X}), \text{FK}_{tcp}(X))$（含 self + cross 四路） | TCP 鎖定，確保夾爪位置不偏 | 5.0 |
| $\mathcal{L}_{MJE}$ | $\sum_t \|\hat{X}^{joints}_t - 3\hat{X}^{joints}_{t-1} + 3\hat{X}^{joints}_{t-2} - \hat{X}^{joints}_{t-3}\|^2$ | 最小 jerk 約束，壓制高頻抖動（僅 joints） | 0.1 |
| $\mathcal{L}_{Content}$ | $\|c_{robot} - c_{human}\|^2$ | 強迫 encoder 忽略 style，只編碼任務意圖 | 1.0 |
| $\mathcal{L}_{Cross}$ | $\frac{1}{2}(\|\text{Dec}(c_r, s_h) - X_h\|_1 + \|\text{Dec}(c_h, s_r) - X_r\|_1)$（**完整 8D**） | 強化 style 解耦：cross-style 需重建全動作含 gripper | 1.0 |
| $\mathcal{L}_{Gripper}$ | $\frac{1}{4}\sum_{\text{4 branches}}\|\hat{X}_{grip} - X_{grip}\|_1$ | **顯式 gripper 保存**：self + cross 所有分支的 gripper 忠實度 | 2.0 |
| $\mathcal{L}_{CrossElbow}$ | $\frac{1}{2}(\text{MSE}(\text{FK}_{elbow}(\hat{X}_{r2h}), \text{FK}_{elbow}(X_h)) + \text{MSE}(\text{FK}_{elbow}(\hat{X}_{h2r}), \text{FK}_{elbow}(X_r)))$ | cross-decoded 肘部應匹配目標 style 的 GT 肘部位置 | 0.5 |

**注意事項**：
- $\mathcal{L}_{Recon}$ **僅對 7D joints**，避免 gripper 誤差梯度干擾 arm style 學習
- $\mathcal{L}_{Gripper}$ **獨立監督 dim 7 (gripper)**，使用較高 λ=2.0 確保精確
- $\mathcal{L}_{Cross}$ 用**完整 8D**（因為 cross-style 仍需精確重建 gripper）
- $\mathcal{L}_{CrossElbow}$ 提供幾何空間監督（Cartesian elbow pos），比 action-space L1 更直接
- $\mathcal{L}_{MJE}$ 只對 7D joints，gripper 是離散指令不需平滑
- $\mathcal{L}_{FK}$ 含四路：self-robot, self-human, cross-r2h, cross-h2r，全部對比 robot TCP
- $\mathcal{L}_{Content}$ 假設 paired demo 時間步對齊；若有偏移可用 soft-DTW
- chunk=8 對 MJE 足夠（需至少 4 個時間步：t, t-1, t-2, t-3）
- 可微分 FK 已實作於 `panda_fk.py`，含 TCP position + elbow (link4) position

### Phase 1 產出物

- **凍結 Encoder**：`Enc(X) → c`，X 為完整 8D（joints + gripper）
- **凍結 Decoder**：`Dec(c, s) → X̂`，輸出完整 8D
- 兩者在 Phase 2 中 `requires_grad=False` 但**不是** `torch.no_grad()`（需讓梯度穿過 decoder 回傳到 Bridge）

### Phase 1 訓練步驟（每個 batch）

```python
# train_step 流程（見 train.py）:
x_robot = batch["robot"]  # (B, 8, 8)  paired robot chunk
x_human = batch["human"]  # (B, 8, 8)  paired human chunk

# 1. Encode both styles（encoder 看完整 8D）
c_robot = model.encode(x_robot)    # (B, 8, D_c)
c_human = model.encode(x_human)    # (B, 8, D_c)

# 2. Self-reconstruction（各自 decode 回自己的 style）
x_hat_robot = model.decode(c_robot, s=0)  # Dec(c_r, s_r)
x_hat_human = model.decode(c_human, s=1)  # Dec(c_h, s_h)

# 3. Cross-reconstruction（用對方的 style 解碼）
x_hat_r2h = model.decode(c_robot, s=1)    # Dec(c_r, s_h) → 應 ≈ x_human
x_hat_h2r = model.decode(c_human, s=0)    # Dec(c_h, s_r) → 應 ≈ x_robot

# 4. 計算 7 項 loss → backward → optimizer.step()
```

> **GeoSCAE 訓練差異**：decode 呼叫需傳入 `x_input` 作為殘差基準與 FK 輸入。
> ```python
> # GeoSCAE decode：
> x_hat_robot = model.decode(c_robot, s=0, x_input=x_robot)
> x_hat_r2h   = model.decode(c_robot, s=1, x_input=x_robot)
> ```

### SCAE → LIBERO Pipeline 整合（Sliding-Window Inference）

訓練完成後，需將 SCAE 模型套用於 LIBERO demo 資料以生成 humanized 動作軌跡。
此步驟已整合進 LIBERO pipeline，透過 `--method scae` 旗標觸發。

#### CLI 使用方式

```bash
# 整套 suite（例如 LIBERO_90）
python A_humanized_libero_suite.py \
    --method scae \
    --scae_checkpoint /path/to/best_model.pt \
    --scae_config /path/to/config.yaml

# 單一 demo
python A_run_pipeline.py --method scae \
    --scae_checkpoint /path/to/best_model.pt \
    --scae_config /path/to/config.yaml \
    --steps scae
```

#### Sliding-Window 推理流程

```
輸入: demo 完整軌跡 τ ∈ R^(T×7) (joints only)
chunk_len = 8 (來自 config)，stride = 1

1. 正規化: τ_norm = normalize(τ)  // Panda joint limits → [-1, 1]
2. Pad: 前後各加 chunk_len-1 個 padding（replicate 邊界值）
3. 展開 sliding windows:
   W = {τ_norm[i : i+chunk_len] for i=0..T-1}  → R^(T × 8 × 7)
4. 拼接 gripper: W_full = concat(W, gripper_chunks) → R^(T × 8 × 8)
5. Batched encode + decode:
   c = model.encode(W_full)               # (T, 8, D_c)
   標準 SCAE: τ̂ = model.decode(c, s=1)   # target style = human
   GeoSCAE:  τ̂ = model.decode(c, s=1, x_input=W_full)  # 需 x_input
6. Hann Window 合併重疊區域:
   每個 window 產出 8 步預測，stride=1 → 大量重疊
   使用 Hann window 加權 → 平滑過渡，消除邊界不連續
   output[t] = Σ_w hann[w] * τ̂[w][t] / Σ_w hann[w]
7. 反正規化: τ_human = denormalize(output)
8. 儲存: humanized.npz → {joint_states_human, gripper_commands_human, ...}
```

#### GeoSCAE 偵測機制

```python
# 由 config 自動偵測模型類型
model_type = getattr(cfg.model, "model_type", "default")
if model_type == "geo":
    from scae.scae_geo import GeoSCAE
    model = GeoSCAE(cfg)
else:
    from scae.scae import SCAE
    model = SCAE(cfg)
```

#### 技術細節

- **CPU-only（並行模式）**：`A_humanized_libero_10.py` 使用 `ProcessPoolExecutor`（30 workers），fork 後 CUDA 不安全，因此 SCAE 模型強制在 CPU。每個 worker 有 `_scae_cache` dict 做 lazy loading。
- **CUDA（單一模式）**：`A_run_pipeline.py` 單 demo 模式支援 CUDA。
- **輸出目錄命名**：`{suite}_humanized_scae/`（非 `{suite}_humanized/`），避免覆蓋既有 preprocess+project 結果。
- **繞過 preprocess**：`method=scae` 時直接從原始 demo → SCAE → humanized.npz，不需要 preprocess 與 project 步驟。

### Phase 1 驗證（Phase 2 之前必做，見 `evaluate.py`）

```python
# 在 LIBERO-Goal 上做 sanity check（跨 task suite 泛化能力）
# evaluate.py 輸出的核心指標：
metrics = {
    # 重建品質（normalized + 弧度空間）
    "recon_l1_robot":      ...,  # self-recon robot L1，應該很小
    "recon_l1_human":      ...,  # self-recon human L1，應該很小
    "cross_l1_r2h":        ...,  # cross robot→human L1
    "cross_l1_h2r":        ...,  # cross human→robot L1

    # Content code 一致性
    "content_mse":         ...,  # c_robot ≈ c_human?
    "content_cosine":      ...,  # 應趨近 1.0

    # TCP 位置誤差（毫米）
    "tcp_rmse_self_robot": ...,  # 應 < 5mm
    "tcp_rmse_cross_r2h":  ...,  # cross-style 也應 < 5mm

    # 肘部位置（style signal 對比）
    "elbow_diff_gt":       ...,  # GT robot vs human 的肘部差異（style 信號大小）
    "elbow_diff_r2h_vs_h": ...,  # cross-decoded 肘部 vs human GT（應很小）

    # Gripper 忠實度
    "gripper_l1_robot":    ...,  # 應趨近 0
    "gripper_l1_cross_r2h": ..., # cross 分支也應趨近 0

    # 平滑度（jerk，越小越平滑）
    "jerk_robot_gt":       ...,  # 原始 robot demo 的 jerk
    "jerk_human_recon":    ...,  # decoded human 比 GT human 更平滑?
    "jerk_cross_r2h":      ...,  # cross-style 的平滑度
}
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
| Phase 1 Encoder 輸入 | 完整 8D（joints + gripper） | Gripper 作為上下文：gripper 狀態影響 approach_blend，encoder 需感知 |
| Phase 1 Decoder 輸出 | 完整 8D（joints + gripper） | FiLM 初始化近 identity → gripper 調制天然趨近零；L_Gripper 顯式守護 |
| Phase 1 Gripper 監督 | joints/gripper 分離監督 | L_Recon 只算 joints；L_Gripper 獨立算 gripper (λ=2.0)，避免梯度干擾 |
| Phase 1 Loss 數量 | 7 項（+Gripper, +CrossElbow） | 原 5 項 + 顯式 gripper 保存 + 幾何肘部監督 |
| Phase 2 human GT | On-the-fly 由 frozen decoder 即時生成 | 不需預先做 IK 轉換；計算成本 <1ms |
| c_gold 來源 | 兩軌都用 robot demo 算 | 保持一致性；content loss 保證 c_reg ≈ c_hum |
| 圖像 | Track A/B 共用 robot 圖像 | TCP 位置一致，視角幾乎相同 |
| Style 表示 | Learned Embedding (dim=32) | 可學習、未來易擴展 |
| Decoder 凍結 | requires_grad=False（非 no_grad） | 讓梯度穿過 decoder 流向 Bridge |
| Track A/B 權重 | α=0.8 / (1-α)=0.2 | Robot GT 可靠主導；pseudo-human GT 輔助學 style |
| D_c | 64（預設） | 平衡表達力與 latent matching 難度 |

---

## 風險與緩解

| 風險 | 嚴重度 | 緩解方法 |
|------|--------|----------|
| Phase 1 AE 在 LIBERO-Goal 泛化失敗 | 🔴 高 | Phase 2 前做 sanity check（`evaluate.py`）；若失敗則 Phase 1 加入更多樣動作資料 |
| IK 產生的 human demo 品質差 | 🔴 高 | Phase 1 前可視化 paired demo；確認 FK/jerk/奇異點 |
| Gripper 信號被 style FiLM 污染 | 🔴 高 | FiLM 初始化近 identity + 顯式 L_Gripper (λ=2.0) + 監控 gripper L1 指標 |
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
