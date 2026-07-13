# Image Storage Process Comparison: Three Pipelines

## Overview
There are three key image storage/loading pipelines in the codebase:
1. **Original dataset generation** (`regenerate_libero_dataset.py`)
2. **Humanized replay with collection** (`A_libero_joint_replay.py` → `A_npz_to_hdf5.py`)
3. **Training & Evaluation** (TFDS builders → training, `run_libero_eval.py` for eval)

---

## 1. Original Dataset Generation (`regenerate_libero_dataset.py`)

**Purpose:** Regenerate LIBERO benchmark dataset by replaying original demonstrations in simulated environment

### Image Collection
- **Source:** Direct environment observations
  - `obs["agentview_image"]` → Third-person camera (256×256 RGB)
  - `obs["robot0_eye_in_hand_image"]` → Wrist camera (256×256 RGB)
- **Storage Location:** Collected in Python lists
- **Transformation:** ✅ **NONE** during collection

### Image Storage to HDF5
**File:** [regenerate_libero_dataset.py](regenerate_libero_dataset.py#L193-L194)
```python
obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
```
- Shape: `(T, 256, 256, 3)` uint8
- Naming: `agentview_rgb`, `eye_in_hand_rgb`
- **Transformation during HDF5 write:** ✅ **NONE**

### Documentation
Lines 5-10 state:
```
- We save image observations at 256x256px resolution (instead of 128x128).
- In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
  180 degrees because we observe that the environments return images that are upside down
  on our platform.
```

---

## 2. Humanized Replay Collection (`A_libero_joint_replay.py`)

**Purpose:** Record humanized manipulations via replay with environment observations collected

### Image Collection
- **Source:** Direct environment observations during playback
  - `obs["agentview_image"]` → Third-person camera
  - `obs["robot0_eye_in_hand_image"]` → Wrist camera
- **Storage:** Appended to Python lists
  - `collected_agentview.append(obs["agentview_image"].copy())`
  - `collected_eye_in_hand.append(obs["robot0_eye_in_hand_image"].copy())`
- **Transformation:** ✅ **NONE** during collection

**File:** [A_libero_joint_replay.py](A_libero_joint_replay.py#L1720-1722)
```python
if not args.no_frontview_obs:
    fv_rgb = render_front_view_rgb(env)
    collected_frontview.append(fv_rgb.copy())
collected_agentview.append(obs["agentview_image"].copy())
collected_eye_in_hand.append(obs["robot0_eye_in_hand_image"].copy())
```

### Image Storage to NPZ
**File:** [A_libero_joint_replay.py](A_libero_joint_replay.py#L2161-2165)
```python
save_dict["agentview_rgb"] = np.stack(collected_agentview)
save_dict["eye_in_hand_rgb"] = np.stack(collected_eye_in_hand)
if not args.no_frontview_obs:
    save_dict["frontview_rgb"] = np.stack(collected_frontview)
```
- Shape: `(T, 256, 256, 3)` uint8
- Naming: `agentview_rgb`, `eye_in_hand_rgb`, `frontview_rgb` (optional)
- **Transformation during NPZ save:** ✅ **NONE**

---

## 3. NPZ → HDF5 Conversion (`A_npz_to_hdf5.py`)

**Purpose:** Convert humanized replay NPZ files to LIBERO HDF5 format for RLDS pipeline

### Image Reading from NPZ
**File:** [A_npz_to_hdf5.py](A_npz_to_hdf5.py#L229-230)
```python
agentview_rgb = sim_data["agentview_rgb"]                  # (T, 256, 256, 3) uint8
eye_in_hand_rgb = sim_data["eye_in_hand_rgb"]              # (T, 256, 256, 3) uint8
```

### Image Storage to HDF5
**File:** [A_npz_to_hdf5.py](A_npz_to_hdf5.py#L279-280)
```python
obs_grp.create_dataset("agentview_rgb", data=agentview_rgb[filtered_idx])
obs_grp.create_dataset("eye_in_hand_rgb", data=eye_in_hand_rgb[filtered_idx])
```
- Shape: `(T, 256, 256, 3)` uint8
- Naming: `agentview_rgb`, `eye_in_hand_rgb`
- **Transformation during HDF5 write:** ✅ **NONE**

**Key observation:** This is a **1:1 pass-through** — images are simply filtered by no-op removal and written as-is.

---

## 4. HDF5 → TFDS Conversion (Dataset Builders)

**Purpose:** Load HDF5 data and convert to TFDS TFRecord format for training

### Image Reading from HDF5
**File:** [LIBERO_10_humanized_dataset_builder.py](../rlds_dataset_builder/LIBERO_10_humanized/LIBERO_10_humanized_dataset_builder.py#L27-28)
```python
images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_rgb"][()]
wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_rgb"][()]
```

### Image Transformation for TFDS
**File:** [LIBERO_10_humanized_dataset_builder.py](../rlds_dataset_builder/LIBERO_10_humanized/LIBERO_10_humanized_dataset_builder.py#L48-49)
```python
'image': images[i][::-1, ::-1],           # rotate 180 degrees
'wrist_image': wrist_images[i][::-1, ::-1],  # rotate 180 degrees
```
- **Transformation:** ✅ **180-degree rotation** applied via `[::-1, ::-1]` (flip rows and columns)
- Applied to both `image` and `wrist_image`

---

## 5. Evaluation (`run_libero_eval.py`)

**Purpose:** Run real-time inference using trained models

### Image Extraction from Environment
**File:** [libero_utils.py](libero_utils.py#L64-74)
```python
def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img

def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img
```

- **Transformation:** ✅ **180-degree rotation** applied during observation preprocessing
- Applied **before** passing to the model

**File:** [run_libero_eval.py](run_libero_eval.py#L349-355)
```python
# Get preprocessed images
img = get_libero_image(obs)
wrist_img = get_libero_wrist_image(obs)

# Resize images to size expected by model
img_resized = resize_image_for_policy(img, resize_size)
wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
```

---

## Summary Table

| Stage | agentview | eye_in_hand | Transformation | Notes |
|-------|-----------|-------------|---|---|
| **Collection** | ✅ From obs | ✅ From obs | None | Raw from environment |
| **NPZ (humanized)** | ✅ Stored | ✅ Stored | None | Direct np.stack |
| **HDF5 (from NPZ)** | ✅ Stored | ✅ Stored | None | Pass-through, no-ops filtered |
| **HDF5 (original gen)** | ✅ Stored | ✅ Stored | None | Generated via replay |
| **TFDS (training)** | ✅ Loaded | ✅ Loaded | **180° rotation** | `[::-1, ::-1]` |
| **Evaluation** | ✅ From obs | ✅ From obs | **180° rotation** | `[::-1, ::-1]` in libero_utils |

---

## Key Findings

### ✅ **CONSISTENT IMAGE STORAGE**
1. **Collection stage:** Both humanized and original pipelines collect raw images identically
   - `agentview_image` → agentview_rgb
   - `robot0_eye_in_hand_image` → eye_in_hand_rgb

2. **HDF5 storage:** Both pipelines store images identically
   - Shape: `(T, 256, 256, 3)` uint8
   - Dataset names: `agentview_rgb`, `eye_in_hand_rgb`
   - No transformation applied

3. **Gripper width computation:** Both use updated formula (after recent fix)
   - Old: `np.mean(gripper_states, axis=1)` ❌ (collapsed to ~0)
   - New: `np.sum(np.abs(gripper_states), axis=1)` ✅

### ✅ **CONSISTENT IMAGE TRANSFORMATION**
1. **Training pipeline:** TFDS builders apply 180° rotation (documented)
2. **Evaluation pipeline:** `libero_utils.py` applies same 180° rotation
   - Both use `img[::-1, ::-1]` syntax

### ✅ **VERIFIED ALIGNMENT**
- Humanized NPZ → HDF5: 1:1 pass-through (no hidden transforms)
- Original replay: Direct HDF5 generation
- Both converge to identical HDF5 structure → TFDS → training

---

## Potential Issues to Monitor

1. **Image orientation:** Rotation happens consistently in both training and eval
   - If model was trained with rotated images, eval must also rotate
   - ✅ Currently verified as matching

2. **Gripper width computation:** All paths now use `sum(abs())` consistently
   - ✅ Fixed in recent updates

3. **Resolution consistency:** All paths use 256×256
   - ✅ Verified in all three pipelines

4. **No hidden transformations in HDF5 write:**
   - A_npz_to_hdf5.py does **not** apply rotation (correct)
   - regenerate_libero_dataset.py does **not** apply rotation (correct)
   - Rotation is applied **after** HDF5 load in TFDS builder (correct)

---

## Conclusion

✅ **The image storage process IS consistent across all three pipelines:**
- Images are collected identically
- Stored to HDF5 identically
- Transformations are applied consistently during TFDS build and evaluation
- Gripper proprioceptive state computation is now unified with the sum(abs()) formula

The 0% success in the failing rollout is **NOT due to image storage inconsistencies**.
