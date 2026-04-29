# 提案方法：Task-Preserving Humanization + TPHC + OpenVLA-OFT

## 完整架構概要

本研究將整體方法拆分為三個連續階段。第一階段以 TH-IK 建立具任務保持性的 humanized paired demonstrations；第二階段以 TPHC 學習從原始 robot 軌跡到 humanized 軌跡的神經轉換模型；第三階段則利用 TPHC 生成之 humanized demonstrations 微調 OpenVLA-OFT，並以任務成功率與 human-likeness 指標進行聯合評估。

整體方法論可概括為：以物理與運動學約束保證 task fidelity，以學習式 humanizer 取代手工 projector 的部署成本，最後將 humanized motion distribution 注入 vision-language-action policy 的訓練過程。

```
Phase 1: HUMANIZATION（TH-IK）
  資料: LIBERO demonstrations + simulation states
  方法: preprocess phase detection + elbow projector constrained optimization
  產出: humanized paired demos + per-frame phase labels

Phase 2: TPHC（Task-Phase-aware Human Correction）
  資料: Phase 1 paired demos（joint + blend + optional image features）
  方法: 多階段訓練（teacher-forced -> blend predictor -> autonomous）
  產出: 可部署的人類化軌跡轉換網路

Phase 3: OpenVLA-OFT 微調與評估
  資料: 由 TPHC 生成的人類化 demo（libero_10/goal/spatial/object）
  方法: 調整 action head 與 proprio pathway，對 humanized trajectories 訓練
  評估: task success rate + human likeness
```

---

## 背景：OpenVLA-OFT 原始架構

### 模型結構

```
攝影機圖片 -> DINOv2 + SigLIP ViT（視覺骨幹）-> patch embeddings
語言指令 -> Llama-2 Tokenizer -> token embeddings
proprio -> ProprioProjector（2層 MLP）-> 1 個 proprio token

完整序列:
[視覺 patches] [proprio token] [文字 tokens] [動作 tokens]

-> Llama-2 Transformer（LoRA）
-> actions hidden states
-> ActionHead（MLPResNet）
-> predicted action chunk (K=8, dim=7 in original LIBERO mode)
```

### 重要常數（LIBERO 設定）

```python
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
  "ACTION_DIM": 7,    # EEF delta (dx, dy, dz, drx, dry, drz, gripper)
    "PROPRIO_DIM": 8,
}
```

---

## Phase 1：HUMANIZATION（TH-IK）

### 1.1 研究目標

本階段之目標為建立一個具任務保持性（task-preserving）的人類化軌跡生成流程。其核心不在於單純改變關節姿態外觀，而在於在保留操控任務成功所需之末端執行器位置、姿態與接觸語義的前提下，對機械手臂冗餘自由度進行人類化重配置。

### 1.2 參考實作

- 主要說明文件：`/home/vsp1323/alex/LIBERO/HUMANIZATION_PIPELINE.md`
- Suite orchestration：`LIBERO/scripts/A_humanized_libero_suite.py`
- Preprocess（phase-aware）：`LIBERO/scripts/A_preprocess_human_demo_17.py`
- Projector（TH-IK solver）：`LIBERO/scripts/A_elbow_projector_25.py`
- Replay：`LIBERO/scripts/A_libero_joint_replay.py`
- Human-likeness evaluation：`LIBERO/scripts/A_human_likeness_evaluate.py`

### 1.3 方法核心

TH-IK 將每一幀的 humanization 問題寫成受限最佳化問題：

$$
\min_q J(q)
$$

subject to

$$
x_{tcp}(q)=x_{demo}, \quad R_{tcp}(q)=R_{demo}\,(critical\ phase), \quad q_{min}\le q\le q_{max}
$$

其中 $J(q)$ 綜合以下目標：

- 肘部 human target 誤差（Soechting-style elbow target）
- 人類關節範圍懲罰
- 時序平滑項（以上一幀解作為 prior）
- 與原始 demonstration posture 的正則化距離

從實作角度看，這一階段的核心求解器並不是學習模型，而是以 SLSQP 為主的 constrained optimization。以目前 `A_elbow_projector_25.py` 的設計而言，每一幀都會根據當前的 demo TCP target、關節上下界、前一幀 warm-start 與 `approach_blend` 對應的姿態約束強度，重新解一個受限非線性最佳化問題。換言之，這是一種 phase-adaptive optimization，而不是參數經資料訓練後直接前向推論的 dynamic learning。

因此，若要在論文中描述本階段，較準確的說法應是：TH-IK 採用「phase-aware constrained optimization」進行任務保持的人類化，而非使用神經網路直接學習 humanization 映射。

### 1.4 Phase-aware preprocessing

本階段之 phase detection 不應僅依賴尾段加權等純時間啟發式規則，而應以任務互動證據建立 manipulation-critical frames。較合理的訊號來源包括：

1. gripper transition（open<->close）
2. articulation state change（門、抽屜、旋鈕）
3. free-body object motion（平移/旋轉）
4. 經時間平滑後得到連續 per-frame blend
5. 輸出可供 Phase 2 使用的 frame-level labels

其中 `A_preprocess_human_demo_17.py` 的角色，是利用 gripper transition、articulation state change、free-body object motion 等物理事件，建立連續的 `approach_blend` 與相關 phase labels。這些訊號本身不是學習得到的，而是由 simulation state 與任務互動證據推導出的 supervision / control signal。

### 1.5 產出資料格式

每個 demonstration 至少應輸出下列資訊：

- `joint_states_demo` (T,7)
- `joint_states_human` (T,7)
- `gripper_commands_demo` (T,)
- `gripper_commands_human` (T,)
- `approach_blend` (T,)
- `artic_blend` (T,) optional
- `obj_pos_blend` (T,) optional
- `phase_labels` (T,) optional（free / transition / critical）
- replay 或 simulation sidecar（含影像與 sim states）

### 1.6 階段性評估

- Replay success（humanized trajectory 是否仍可完成任務）
- TCP position/orientation error
- 時序平滑與 jerk 指標
- Human-likeness 指標（肘部幾何、姿態自然度、關節範圍違規率）

---

## Phase 2：TPHC

### 2.1 研究目標

本階段目標為學習一個由原始 robot 軌跡到 humanized 軌跡的映射模型，即以神經網路近似 TH-IK 所實現的 phase-aware humanization 過程，降低傳統 projector 在部署時的運算成本，同時保留 task-relevant 幾何約束。

若要從論文貢獻的角度濃縮，本階段的核心主張可表述為：

1. humanization 不應是固定強度的姿態轉換，而應依任務階段動態調整 human-likeness；
2. 這種調整不再由手工規則直接驅動，而是由可部署觀測訊號預測出的 phase signal 所調控；
3. 模型學習的是「在不破壞 task completion 的前提下，於不同階段施加不同程度的人類化修正」。

### 2.2 參考實作

- 專案根目錄：`/home/vsp1323/alex/TPHC`
- 說明文件：`/home/vsp1323/alex/TPHC/README.md`
- 主要網路：`TPHC/tphc/models/tphc_net.py`
- 訓練腳本：
  - `TPHC/tphc/training/train_stage1.py`
  - `TPHC/tphc/training/train_stage2.py`
  - `TPHC/tphc/training/train_stage3.py`

### 2.3 模型形式

TPHC 可被視為一個 task-phase-aware 的殘差轉換器：

$$
\hat q_{hum} = q_{raw} + \hat g \odot \Delta q
$$

- `q_raw`：原始關節 chunk
- `geo_feats`：幾何特徵（FK keypoints、elbow geometry、joint-limit related features）
- `blend`：phase conditioning（teacher signal 或 predictor 輸出）
- `images`：Stage 2 / Stage 3 可選的視覺輸入（agentview + eye_in_hand）

這裡的 `approach_blend` 不宜被簡化理解為一般的 smoothing coefficient，而更接近「task-criticality-conditioned humanization control signal」。當 `blend` 較高時，模型傾向保守地貼近原始 task-preserving posture；當 `blend` 較低時，模型允許較大的 human-like correction。也就是說，TPHC 實際上是在學習一個由 phase signal 調控的人類化強度分配機制。

### 2.4 三階段訓練流程

1. **Stage 1（teacher-forced humanizer）**
   - 使用 ground-truth blend 作為 conditioning
   - 任務定義為 `GT blend + q_raw + geo_feats -> q_hum`
   - 目的在於先學會穩定且可重現的軌跡轉換

2. **Stage 2（predictor-only）**
   - 訓練 phase/blend predictor 分支（Plan A 或 Plan B1）
   - 任務定義為 `q_raw + geo_feats + images -> blend_pred`
   - 目的在於從可部署資訊中恢復 phase conditioning

3. **Stage 3（autonomous end-to-end）**
   - 以實際部署路徑進行 end-to-end 訓練：`q_raw + images -> blend_pred -> q_hum`
   - 融合 Stage 1 與 Stage 2 權重，以提升整體穩定性與可部署性

從目前 `TPHC/tphc/models/tphc_net.py` 的實作來看，Stage 3 的關鍵不只是「有 vision」，而是模型會先由 `q_raw + gripper + geo_feats + images` 預測 `blend_pred`，再以該 phase signal 調控 residual humanization。故若要在論文中描述本方法，較精確的措辭應是「observation-conditioned phase-aware humanization」，而不是籠統地稱為 dynamic learning。

若要進一步強調 contribution，可以寫成：TPHC 以可觀測之 motion-geometry-vision cues 自動估計 task phase，並依此動態控制 human-likeness level，以兼顧 human-like posture 與 task completeness。

### 2.5 訓練資料格式（重點）

核心資料來自 Phase 1 humanized outputs：

- `joint_states_demo`、`joint_states_human`
- `approach_blend`
- `gripper_commands_demo`
- `artic_blend` / `obj_pos_blend`（Plan B1）
- `original_sim.npz` 的可部署觀測影像（Stage 2/3 predictor input）
- `humanized_sim.npz` 可作為 GT replay 檢查與分析 sidecar，但不是 predictor 的主要輸入來源

### 2.6 主要損失項（示意）

- `L_recon`：關節重建
- `L_fk_tcp`：TCP FK 誤差
- `L_mje`：平滑度
- `L_elbow`：肘部幾何一致
- `L_blend`：phase/blend prediction

### 2.7 TH-IK 與 TPHC 的方法論差異

這一節在論文中建議明確寫出，因為兩者雖然都會使用 `approach_blend`，但其本質完全不同。

1. **TH-IK / SLSQP projector**
  - 本質：逐幀 constrained optimization
  - `approach_blend` 的作用：調整約束強度，特別是 EEF orientation constraint 的保留程度
  - 依賴：顯式目標函數、物理/運動學約束、每幀數值求解
  - 優點：可解釋、可直接保證 task-space constraint
  - 代價：部署時計算成本較高，且規則設計與 solver 調參較繁瑣

2. **TPHC**
  - 本質：學習式 residual converter
  - `approach_blend` 的作用：作為 phase-conditioned humanization gate / modulation signal
  - 依賴：資料學習得到的映射、motion + geometry + visual observation
  - 優點：部署時為單次前向推論，較易擴展至大規模資料生成
  - 代價：需承受 train/inference mismatch、phase prediction error、distribution shift 等學習風險

因此，若直接回答「SLSQP 是不是一種 dynamic learning」，答案是否定的。SLSQP 是一種數值最佳化方法；它可以隨每幀條件動態求解，但不屬於 data-driven learning。相對地，TPHC 才是學習式方法，且其動態性來自 observation-conditioned phase prediction 與 phase-aware modulation。

---

## Phase 3：TPHC 生成資料、OpenVLA-OFT 微調與多維評估

### 3.1 研究目標

本階段目標為利用 TPHC 批次生成人類化 demonstrations，並以此作為 supervision 微調 OpenVLA-OFT，使策略在維持任務成功率的同時，學得更接近 humanized motion distribution 的 action 與 proprio 表徵。

若從 thesis story 的角度整合三個階段，本研究的主線可被描述為：先以 constrained optimization 建立高品質 task-preserving humanized supervision，再以 phase-aware neural converter 學會可部署的人類化轉換，最後將該 humanized motion distribution 注入 vision-language-action policy，驗證其對 task success 與 human likeness 的聯合影響。

### 3.2 目標資料集

- `libero_10`
- `libero_goal`
- `libero_spatial`
- `libero_object`

對每一個 suite，應建構下列資料：

- humanized joint trajectories
- 對應 gripper commands
- 必要 replay/sim metadata
- （可選）phase 相關預測輸出，用於後續誤差歸因與可視化分析

### 3.3 OpenVLA-OFT 具體改造項目

本階段不僅是更換 supervision source，而是需要對 OpenVLA-OFT 的 action 與 proprio 介面做明確調整，使其訓練目標與 humanized demonstrations 保持一致。

#### 3.3.1 資料載入與 supervision 定義

1. 在 dataset loader 中加入 humanized trajectory 版本的 action labels。
2. 將每個 training sample 的未來 action chunk 由原始 robot demo 改為對應的人類化 joint trajectory。
3. 保留 gripper supervision，但使其與 humanized joints 對齊為同一時間窗。
4. 若同時保留原始 demo，可額外建立 `robot/humanized` 雙版本 supervision 以支援 ablation。

#### 3.3.2 Action 表徵與 action head 修改

此處的修改不應被理解為僅更換 regression target。由於原始 OpenVLA-OFT 在 LIBERO 設定下採用的是 7D EEF delta control，而目前的人類化版本改為 8D joint-position control，因此實際上是整個 action representation 與下游控制介面的同步切換。

1. 維持 `NUM_ACTIONS_CHUNK = 8` 不變，但 action semantics 由「7D EEF delta action」改為「8D absolute joint-position action」。
2. 將 `ACTION_DIM` 由原始 LIBERO 模式的 7 改為 humanized / joint-control 模式的 8，對應 `7 joints + 1 gripper`。此修改不是僅存在於 loss target，而是會連動 action token 對應的 hidden-state reshape、action head 建構，以及 diffusion 分支的 tensor shape。
3. 將 action head 的 regression target 由原始 robot EEF action 改為 humanized joint action；在實作上，`get_action_head(...)` 已以全域 `ACTION_DIM` 建構 head，因此 head 的輸入展平尺寸與輸出維度都會跟著切換。
4. 資料集本身亦需先完成 action representation 轉換，而不是直接拿原始 HDF5 使用。現有流程已透過 `A_convert_libero_to_joint.py` 將原始 `7D EEF delta` demonstrations 轉成 `8D joint position actions`，並同步建立 `8D proprioceptive state`。
5. rollout / evaluation controller 也必須一併切換。現有 `run_libero_eval.py` 與 `libero_utils.py` 已加入 `use_joint_pos` 路徑，使用 `JOINT_POSITION` controller，並將 policy 輸出的 absolute joint targets 轉為控制器實際執行所需的 normalized delta command。也就是說，policy output 的語義、控制器輸入的語義、以及環境執行方式三者都已一併調整。
6. 若現有 action normalization 以原始 robot demonstrations 的統計量建立，則需重新檢查 humanized demonstrations 的分佈是否仍適用原 normalization；若偏移明顯，需重估 normalization statistics。
7. 訓練與推論時皆應確保 action head 的輸出語義一致，即輸出的是「humanized joint command」，而非原始 robot EEF delta command。

#### 3.3.3 Proprio pathway 修改

此處的修改同樣不只是「把資料換掉」而已。雖然 `PROPRIO_DIM` 在原始 LIBERO 與 humanized / joint-control 模式下皆為 8，但其語義已經改變：原始模式為 EEF-based proprio，humanized / joint-control 模式則改為 joint-based proprio。因此需要將 proprio representation、projector 輸入語義、資料來源與 rollout 時的狀態更新方式一起對齊。

1. 將 proprio 輸入由原始 LIBERO 模式的 `EEF pos (3) + EEF axis-angle (3) + gripper qpos (2)`，改為 humanized / joint-control 模式的 `joint_pos (7) + gripper_width (1)`。
2. 雖然 `PROPRIO_DIM = 8` 維度本身未變，但 `ProprioProjector` 實際接收的特徵語義已經改變，因此不能將其視為僅有 label change；其投影對象已由 task-space state 改為 joint-space state。
3. 訓練資料端需確保 dataset 中的 `state` / `proprio` 欄位已切換為 joint-based proprio。現有 joint-conversion 流程已在 `A_convert_libero_to_joint.py` 中同步產生 `8D proprioceptive state`，而非沿用原始 EEF state。
4. 推論與 rollout 端亦需保持相同語義。現有 `run_libero_eval.py` 已依 `use_joint_pos` 分支切換 observation preparation：在 joint-control 模式下，policy 看到的是 `robot0_joint_pos + gripper_width`，而非 `robot0_eef_pos + quat2axisangle + gripper_qpos`。
5. 因此，train / inference mismatch 的主要風險不再只是數值分佈偏移，而是 state semantics mismatch；若訓練時使用 joint-based proprio，評估時仍餵入 EEF-based proprio，則雖然維度相同，模型實際接收到的狀態空間已完全不同。
6. 若 rollout 過程由 policy 自身預測動作並更新 proprio，則評估時必須使用與訓練一致的 closed-loop joint-state update；否則 policy 會在未見過的 state transition 下工作。
7. 若需要做 ablation，可額外保留一組「robot proprio + humanized action label」或「EEF proprio + joint action label」的對照設定，用於分析表徵切換對 performance 的影響，但此設定應被視為對照組，而非主方法。

#### 3.3.4 訓練程式修改面

從目前 repo 的實作來看，訓練程式修改面應被理解為「整個 humanized / joint-control training stack 的切換」，而不只是 `finetune.py` 內更換 supervision。至少包含下列層級：

1. **常數與模式切換層**：`prismatic/vla/constants.py`
  - 區分 `LIBERO_ORIGINAL_CONSTANTS` 與 `LIBERO_HUMANIZED_CONSTANTS`。
  - 自動切換 `ACTION_DIM`、`PROPRIO_DIM`、`NUM_ACTIONS_CHUNK` 與 normalization mode。
  - 這一層決定了後續 action/proprio tensor shape 與語義。

2. **訓練入口與 run configuration 層**：`vla-scripts/finetune.py`
  - 依 `ROBOT_PLATFORM` 決定 run mode，例如 `--joint_ctrl` / `--eef_ctrl`。
  - 在 batch 組裝與 loss 計算時，使用新的 `ACTION_DIM` 與 humanized labels。
  - action hidden-state reshape 直接依賴 `NUM_ACTIONS_CHUNK * ACTION_DIM`，因此 joint-control 模式不是單純換 target，而是整個 action decoding tensor layout 都跟著改變。

3. **模型前向與 proprio token 處理層**：`prismatic/extern/hf/modeling_prismatic.py`
  - `_process_proprio_features(...)` 本身的投影流程未大改，但輸入 `proprio` 的實際語義已從 EEF state 切換為 joint state。
  - action / noisy-action 的 reshape 也依賴全域 `ACTION_DIM`，因此此檔案實際參與了 joint-control representation 的切換。

4. **action head / projector 建構層**：`experiments/robot/openvla_utils.py`、`prismatic/models/action_heads.py`
  - `get_action_head(...)` 以全域 `ACTION_DIM` 建立 `L1RegressionActionHead` 或 `DiffusionActionHead`。
  - `get_proprio_projector(...)` 以全域 `PROPRIO_DIM` 建立 proprio projector。
  - 因此 head/projector 雖沿用同一類別，但其實例化後的輸入輸出語義與 shape 已隨模式改變。

5. **資料轉換與 dataset 準備層**：`experiments/robot/libero/A_convert_libero_to_joint.py`
  - 將原始 `7D EEF delta` HDF5 demonstrations 轉換為 `8D joint position actions`。
  - 同步產生 `8D joint-based proprio state`。
  - 這一步是 humanized / joint-control 訓練得以成立的前置條件，而非可省略的 data preprocessing 細節。

6. **rollout / evaluation / controller 對齊層**：`experiments/robot/libero/run_libero_eval.py`、`experiments/robot/libero/libero_utils.py`
  - 評估時需切換 `use_joint_pos`，使用 `JOINT_POSITION` controller。
  - 將模型輸出的 absolute joint targets 轉為 controller 可執行的 normalized delta command。
  - observation preparation 與 dummy action 也需配合 joint-control 模式切換。

7. **訓練與評估一致性原則**
  - 若訓練時採用 joint-based actions + joint-based proprio，則 evaluation、rollout、success verification 與資料重建流程都必須使用同一表示法。
  - 否則即使 backbone 與 head 類型未變，整體 policy 行為仍會因 action/state semantics mismatch 而失效。

#### 3.3.5 訓練策略建議

1. 先於 `libero_10` 進行 warm-up，確認資料格式與訓練穩定性。
2. 再擴展至 `libero_goal`、`libero_spatial`、`libero_object`。
3. 若 humanized distribution 較原始 robot distribution 更平滑，可比較不同 loss 權重下的收斂性。
4. 建議保留原始 robot policy 作為 baseline，與 humanized-data fine-tuned policy 做直接比較。

### 3.4 評估協議（必做）

#### A. 任務成功率（Task Performance）

對每個 suite 報告：

- Success rate（%）
- 任務級別平均與標準差
- 各 task 成功數 / 失敗數

#### B. Human-likeness Evaluation

至少包含：

- 肘部姿態幾何指標（與 TH-IK/TPHC target 的一致性）
- 關節動作平滑度（jerk / acceleration）
- 關節範圍違規率
- 末端執行器姿態保真（critical phases）

#### C. 綜合報告

最終需同時呈現：

- `Success`（任務是否完成）
- `Human-likeness`（動作是否更像人）

避免只優化其一。

---

## 流程摘要

| 面向 | 內容 |
|------|------|
| Phase 1 | TH-IK HUMANIZATION pipeline |
| Phase 2 | TPHC 多階段訓練 |
| Phase 3 | TPHC 生成資料後微調 OpenVLA-OFT 並評估 |
| 核心 supervision | paired trajectory + phase labels |
| 主要風險 | phase detection / replay fidelity / suite transfer |

---

## 建議實驗順序

1. 完成 Phase 1（TH-IK）於 LIBERO-90，檢查 replay success 與 human-likeness
2. 用同一資料訓練 TPHC Stage1/2/3，先做 chunk-level 再做 long-demo evaluation
3. 用 TPHC 生成 `libero_10/goal/spatial/object` humanized demos
4. 基於 humanized demos 微調 OpenVLA-OFT（含 action/proprio 對齊）
5. 在四個 suite 統一評估 success rate + human likeness
6. 針對失敗任務回查 phase labels / blend prediction 做誤差分析

---

## 論文定位與題目草案

### 論文主貢獻的建議表述

若要把 `approach_blend` 納入論文主貢獻，建議不要只說「dynamic human likeness」，而是更精確地寫成以下其中一種：

1. task-phase-aware humanization
2. observation-conditioned human-likeness modulation
3. dynamic human-likeness control for task-preserving motion conversion

其中第三種最接近你想表達的意思，但論文中最好再補一句：這個 dynamic control 並不是泛指任意時間變化，而是由可觀測的 phase evidence（motion、geometry、vision）所驅動，目的是在不同操作階段調整 humanization 強度，以維持 task completeness。

### 題目草案

以下題目相對貼近目前實作，也比較像論文標題語氣：

1. Task-Phase-Aware Humanization of Robot Demonstrations for Vision-Language-Action Policy Adaptation
2. Task-Preserving Humanized Conversion of Robot Demonstrations via Observation-Conditioned Human-Likeness Modulation
3. From Constrained Humanization to Policy Adaptation: Learning Task-Preserving Humanized Robot Demonstrations for OpenVLA-OFT
4. Learning Observation-Conditioned Humanized Robot Demonstrations with Dynamic Human-Likeness Control

若你要把 OpenVLA-OFT 也放進標題，我會優先考慮第 1 或第 3。若你要把「動態控制 human-likeness」作為主要 novelty，我會優先考慮第 2。

### 摘要草案

本研究探討如何在不破壞任務完成性的前提下，將機器人示範軌跡轉換為更具人類動作特徵的操作軌跡，並進一步用於 vision-language-action policy 的微調。傳統 robot demonstrations 雖可有效完成 manipulation tasks，但其關節姿態與運動分佈通常缺乏人類操作的自然性，限制了 human-like motion 建模與後續策略學習的表徵品質。為此，本研究提出一個三階段框架。首先，我們以 task-preserving humanization pipeline 建立高品質 paired demonstrations，透過 phase-aware preprocessing 偵測 gripper transition、articulation change 與 object motion 等 manipulation-critical events，並結合受限最佳化的 TH-IK projector，在保留末端執行器 task-space 關鍵約束下生成人類化 joint trajectories。其次，我們提出 TPHC，一個 task-phase-aware 的殘差式神經轉換模型，用以近似傳統 projector 的人類化過程。TPHC 並非採用固定強度的人類化策略，而是根據 motion、geometry 與 visual observations 預測 phase-conditioned control signal，以動態調節 human-likeness level，使模型在 task-critical phases 保持較高的 task fidelity，在較自由的操作階段則施加較強的人類化修正。最後，我們將 TPHC 生成的人類化 demonstrations 用於 OpenVLA-OFT 微調，並對 action representation、proprioceptive pathway 與 rollout controller 進行 joint-control 對齊。實驗將於多個 LIBERO benchmark suites 上評估任務成功率與 human-likeness 指標，以驗證 observation-conditioned humanization 對 policy adaptation 的效益。 

---

## 附錄：最小可行流程（MVP）

```bash
# Phase 1: TH-IK suite humanization
python LIBERO/scripts/A_humanized_libero_suite.py --suite libero_90

# Phase 2: TPHC
python -m tphc.training.train_stage1 --config TPHC/configs/stage1.yaml
python -m tphc.training.train_stage2 --config TPHC/configs/stage2_plan_a.yaml
python -m tphc.training.train_stage3 --config TPHC/configs/stage3_plan_a.yaml

# Phase 3: 用 TPHC 生成各 suite humanized demos，再微調 OpenVLA-OFT
# （依 openvla-oft_human 現有 finetune pipeline 接入 humanized data）
```

