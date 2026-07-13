# Send Dataset / Checkpoint Command Reference

Local-side paths are relative to the `openvla-oft_human/` repo root (run from inside that folder). Remote-side paths are on the VCP cluster (`vcpuser@vcppx.ntut.edu.tw`) and stay absolute — that's a different machine, unaffected by local folder layout. SSH ports vary per allocation; substitute the port you were given.

`<method>` = `th_ik` / `pure_ik` / `liu_ik` / `hrr_ik` / `original` (matches the `modified_libero_rlds/<method>/` and `runs/<method>/` layout from `docs/command/Make training dataset.md` and `docs/command/finetune command.md`).

## 1. Send dataset: local → VCP

```bash
rsync -rlptDvz --progress --no-g -e "ssh -p <port>" \
  modified_libero_rlds/<method>/<suite>_<humanized|joint>_no_noops \
  vcpuser@vcppx.ntut.edu.tw:~/netdrive/Workspace/Alex/openvla-oft_human/modified_libero_rlds/<method>/
```

Multiple suites in one call: list several local source paths before the remote destination, e.g.

```bash
rsync -rlptDvz --progress --no-g -e "ssh -p <port>" \
  modified_libero_rlds/<method>/libero_10_humanized_no_noops \
  modified_libero_rlds/<method>/libero_spatial_humanized_no_noops \
  modified_libero_rlds/<method>/libero_goal_humanized_no_noops \
  modified_libero_rlds/<method>/libero_object_humanized_no_noops \
  vcpuser@vcppx.ntut.edu.tw:~/netdrive/Workspace/Alex/openvla-oft_human/modified_libero_rlds/<method>/
```

Whole-directory sync (all suites/methods at once): `rsync ... modified_libero_rlds/ vcpuser@...:~/netdrive/Workspace/Alex/openvla-oft_human/modified_libero_rlds/`.

## 2. Send checkpoint: VCP → local

```bash
mkdir -p runs/<method>/openvla-oft_<humanized|joint>_<suite>

# whole run dir
rsync -rlptDvz --progress --no-g -e "ssh -p <port>" \
  vcpuser@vcppx.ntut.edu.tw:~/netdrive/Workspace/Alex/openvla-oft_human/runs/<method>/openvla-oft_<humanized|joint>_<suite>/ \
  runs/<method>/openvla-oft_<humanized|joint>_<suite>/

# only one checkpoint subdir (faster — skips intermediate/older checkpoints)
rsync -rlptDvz --progress --no-g -e "ssh -p <port>" \
  vcpuser@vcppx.ntut.edu.tw:~/netdrive/Workspace/Alex/openvla-oft_human/runs/<method>/openvla-oft_<humanized|joint>_<suite>/<checkpoint_dir_name> \
  runs/<method>/openvla-oft_<humanized|joint>_<suite>/
```

## 3. Send checkpoint: local → VCP

```bash
ssh -p <port> vcpuser@vcppx.ntut.edu.tw "mkdir -p ~/netdrive/Workspace/Alex/openvla-oft_human/runs/<method>/openvla-oft_<humanized|joint>_<suite>"

rsync -rlptDvz --progress --no-g -e "ssh -p <port>" \
  runs/<method>/openvla-oft_<humanized|joint>_<suite>/<checkpoint_dir_name> \
  vcpuser@vcppx.ntut.edu.tw:~/netdrive/Workspace/Alex/openvla-oft_human/runs/<method>/openvla-oft_<humanized|joint>_<suite>/
```

`original` runs (no humanization) omit the `<method>` path segment historically used stricter naming (`openvla-oft_joint_<suite>` / `openvla-oft_libero_4tasks` without a leading method dir in older transfers) — prefer the `runs/original/...` form above for anything new so it lines up with `finetune_libero_from_rlds.sh`'s output.
