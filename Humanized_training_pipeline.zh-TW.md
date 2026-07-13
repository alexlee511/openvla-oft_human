# 人性化訓練流程（openvla-oft_human）— 繁體中文版

> 本文為 `Humanized_training_pipeline.md`（正式版，英文）的繁體中文翻譯，內容應保持同步。若兩者有出入，以英文版為準。

本文說明這個 repo 裡的訓練／評估流程。輸入資料來自姊妹 repo `LIBERO-humanized`（`humanized.npz`／`humanized_sim.npz` 是怎麼跨五種比較方法——Original、Pure-IK、Liu-IK、HRR-IK、TH-IK/本研究方法——與各項消融實驗產生的，請見 `LIBERO-humanized/HUMANIZATION_PIPELINE.md`）。本文只涵蓋資料進到 `openvla-oft_human` 之後發生的事。

```text
humanized.npz / humanized_sim.npz（來自 LIBERO-humanized，依方法/suite/任務分）
    │
    ▼
資料集轉換 → RLDS
    │
    ▼
OpenVLA-7B LoRA 微調（關節控制）
    │
    ▼
在 LIBERO 模擬環境中評估政策
    │
    ▼
rollout 人性化程度評分
```

## 一、資料集轉換

| 檔案 | 角色 |
|---|---|
| `experiments/robot/libero/A_npz_to_hdf5.py` | 把回放輸出（`A_libero_joint_replay.py --collect_obs` 產生的 `humanized_sim.npz`／`original_sim.npz`）打包成 HDF5。 |
| `rlds_dataset_builder/LIBERO_{10,Goal,Object,Spatial}_{humanized,joint}/` | 把 HDF5 轉成 RLDS TFRecord 資料集的 TFDS builder（每個 suite × {humanized, joint} 各一個目錄）。 |
| `scripts/rebuild_libero_rlds_from_npz.sh` | 端到端包裝腳本：NPZ → HDF5 → RLDS，一次處理一組 `<humanized\|original-joint> <suite> [method]`——`[method]` 是 `pure-ik`／`liu-ik`／`hrr-ik`／`th-ik`（或消融標籤如 `th-ik_cs-cap`），對應 `LIBERO-humanized/scripts/A_humanized_libero_suite.py` 產生的輸出目錄命名。完整指令集見 `docs/command/Make training dataset.md`。 |

每個方法——包括 `th-ik`（本研究方法）和 `original-joint` 基準——都會在 `modified_libero_rlds/` 底下有自己的子目錄：`modified_libero_rlds/th_ik/`、`.../pure_ik/`、`.../liu_ik/`、`.../hrr_ik/`、`.../original/`。這就是 `finetune.py` 之後用來分辨自己在訓練哪個方法/基準的依據；不同方法的資料集不會互相碰撞。

## 二、微調

`vla-scripts/finetune.py` 對 OpenVLA-7B 做 LoRA 微調，內容接近原版 upstream openvla-oft 檔案；原始 vs 人性化、EEF vs 關節控制**不是** runtime 參數，完全由 `--data_root_dir`／`--dataset_name` 指向哪個 RLDS 資料集決定。

⚠️ **這個環境裡實際上沒有真的跑過**：下面標準設定至少需要 **~80GB VRAM**（A100-80GB／H100 等級）。`scripts/finetune_libero_from_rlds.sh` 是給未來有合適硬體時使用的參考包裝腳本——目前只是有文件記錄，並未在這個環境驗證過/ 實際執行過。

`bash scripts/finetune_libero_from_rlds.sh <humanized|original-joint> <suite> [method]` 會用跟上面 `rebuild_libero_rlds_from_npz.sh` 相同的子目錄對應規則，自動算出 `--data_root_dir`／`--run_root_dir`，確保資料集跟訓練輸出永遠對得起來。也可以直接手動呼叫——完整的每方法/每 suite 指令集與從 checkpoint 恢復訓練的做法見 `docs/command/finetune command.md`；若在不支援 bfloat16 的 GPU（T4/V100）上訓練，需要先跑 `vla-scripts/patch_dtype.py`。

## 三、政策評估

| 檔案 | 角色 |
|---|---|
| `experiments/robot/libero/A_generate_humanized_initial_poses.py` | 預先計算人性化的初始機器人姿態（人性化示範的起始姿態可能跟原始 suite 不同），寫出 `A_humanized_initial_poses.py`。 |
| `experiments/robot/libero/run_libero_eval.py` | 核心評測器——雖然檔名沒有 `A_` 前綴，卻是整個 fork 裡改動最多的檔案。載入微調後的 checkpoint，在 LIBERO 模擬環境中做閉環 rollout，記錄到 wandb，儲存每個 episode 的 rollout NPZ 檔。關鍵參數：`--use_joint_pos`（8D 絕對關節控制器）、`--use_humanized_initial_pose`（人性化關節訓練會自動開啟）、`--joint_substeps`、`--joint_Kp_overshoot`、`--unnorm_key`（會模糊比對 `_humanized_no_noops` 這類資料集後綴）。 |

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint runs/th_ik/openvla-oft_humanized_libero_10/<checkpoint_dir> \
  --task_suite_name libero_10 \
  --use_l1_regression true --use_proprio true --use_joint_pos true \
  --use_humanized_initial_pose true \
  --num_images_in_input 2 --center_crop true \
  --num_open_loop_steps 8 --num_trials_per_task 10 \
  --joint_substeps 1 --joint_Kp_overshoot 3.5
```

完整記錄的指令集（原始 EEF／原始關節／各個人性化方法）見 `docs/command/libero evaluate command.md`。

## 四、Rollout 人性化程度評分

`experiments/robot/libero/A_vla_rollout_hl.py` 為上面產生的 rollout NPZ 檔評分。它沒有重新實作指標，而是把 `../LIBERO-humanized/scripts` 加進 `sys.path`，直接從姊妹 repo 的 `A_human_likeness_evaluate.py` import `evaluate_trajectory`／`METRIC_KEYS`／`WEIGHTS`，所以兩邊的指標定義（`HJL`、`MJE`、`SOAq`、`SOAx`、`EEA`、`UNIFIED`）完全一致。

```bash
python experiments/robot/libero/A_vla_rollout_hl.py \
  --rollout_dir ./rollouts/libero_10/<checkpoint_name>/<datetime>/ \
  --suite libero_10
```

記錄範例見 `docs/command/human likeness eval command.md`。

## 五、輔助分析工具

不在上面的自動流程裡，而是事後對其輸出做分析用——全部放在 `experiments/robot/libero/analysis/` 底下：`A_eval_model_on_train_data.py`（比對模型預測跟訓練集真值）、`A_joint_distribution_diagnostics.py`、`A_rollout_controller_tracking.py`、`A_rollout_tracking_and_trainset_analysis.py`、`A_task_eval_summary.py`。`merge_lora_weights_and_save.py` 則是在 `--merge_lora_during_training` 沒開的情況下，把純 LoRA checkpoint 融合進底模型用的工具。
