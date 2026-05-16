"""Generate stored humanized initial arm poses from LIBERO benchmark initial states."""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import mujoco
import numpy as np
from libero.libero import benchmark
from scipy.optimize import minimize as scipy_minimize

from experiments.robot.libero.libero_utils import get_libero_env


DEFAULT_OUTPUT_PATH = Path(__file__).with_name("A_humanized_initial_poses.py")
DEFAULT_SUITE_NAMES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
FOUR_TASK_SUITE_NAMES = ["libero_10", "libero_spatial", "libero_goal", "libero_object"]
SUITE_NAME_ALIASES = {"libero_4_tasks": FOUR_TASK_SUITE_NAMES}
SUITE_ROOT_TOKENS = {
    "libero_10": "10",
    "libero_spatial": "spatial",
    "libero_goal": "goal",
    "libero_object": "object",
}

MJ_NAME2ID = getattr(mujoco, "mj_name2id")
MJ_FWD_POSITION = getattr(mujoco, "mj_fwdPosition")
MJT_OBJ: Any = getattr(mujoco, "mjtObj")

L_UPPER = np.sqrt(0.316**2 + 0.0825**2)
L_FORE = np.sqrt(0.384**2 + 0.0825**2)
L_TOTAL = L_UPPER + L_FORE

HUMAN_LIMITS_DEG = [
    (-50, 130),
    (0, 98),
    (0, 163),
    (-150, -7),
    (10, 163),
    (100, 212),
]
HUMAN_LIMITS_RAD = [(np.deg2rad(lo), np.deg2rad(hi)) for lo, hi in HUMAN_LIMITS_DEG]


def _muj2med(vector):
    return np.array([-vector[1], vector[0], vector[2]], dtype=np.float64)


def _med2muj(vector):
    return np.array([vector[1], -vector[0], vector[2]], dtype=np.float64)


@dataclass
class SoechtingMap:
    a0: float = -4.0
    a1: float = 1.10
    a2: float = 0.90
    c0: float = 13.2
    c1: float = 0.86
    c2: float = 0.11

    def __call__(self, r_cm, phi_deg, chi_deg):
        q_eu = self.a0 + self.a1 * r_cm + self.a2 * phi_deg
        q_yu = self.c0 + self.c1 * chi_deg + self.c2 * phi_deg
        return q_eu, q_yu


class HumanElbowProjector:
    """Copied minimal projector logic from A_elbow_projector_25.py for init-pose generation."""

    def __init__(
        self,
        arm_joint_names: Sequence[str],
        shoulder_body: str = "robot0_link2",
        elbow_body: str = "robot0_link4",
        wrist_body: str = "robot0_link6",
        tcp_body: str = "gripper0_eef",
        upper_arm_len_m: float = L_UPPER,
        forearm_len_m: float = L_FORE,
        map_fn: Optional[SoechtingMap] = None,
        slsqp_maxiter: int = 100,
        slsqp_ftol: float = 1e-9,
        w_elbow: float = 36.0,
        w_limit: float = 1.0,
        w_posture: float = 0.05,
        w_temporal: float = 30.0,
        ori_constraint_max: float = 1.0,
        # Interface compatibility with A_elbow_projector_25.py
        # (accepted for parity; not used by this SLSQP objective path)
        w_ori: float = 0.0,
        approach_ori_boost: float = 14.0,
        approach_elbow_damp: float = 0.95,
        w_link7: float = 6.0,
    ):
        self.arm_joint_names = list(arm_joint_names)
        self.shoulder_body = shoulder_body
        self.elbow_body = elbow_body
        self.wrist_body = wrist_body
        self.tcp_body = tcp_body
        self.L1 = upper_arm_len_m
        self.L2 = forearm_len_m
        self.map_fn = map_fn or SoechtingMap()
        self.slsqp_maxiter = slsqp_maxiter
        self.slsqp_ftol = slsqp_ftol
        self.w_elbow = w_elbow
        self.w_limit = w_limit
        self.w_posture = w_posture
        self.w_temporal = w_temporal
        self.ori_constraint_max = ori_constraint_max
        self.w_ori = w_ori
        self.approach_ori_boost = approach_ori_boost
        self.approach_elbow_damp = approach_elbow_damp
        self.w_link7 = w_link7
        self._prev_q = None
        self._prev_ref = None

    def bind(self, sim):
        model = sim.model
        data = sim.data
        if hasattr(model, "_model"):
            model = model._model
        if hasattr(data, "_data"):
            data = data._data
        self.model = model
        self.data = data

        self.qpos_adrs = []
        self.robot_lo = []
        self.robot_hi = []
        for joint_name in self.arm_joint_names:
            joint_id = MJ_NAME2ID(model, MJT_OBJ.mjOBJ_JOINT, joint_name)
            self.qpos_adrs.append(model.jnt_qposadr[joint_id])
            self.robot_lo.append(float(model.jnt_range[joint_id, 0]))
            self.robot_hi.append(float(model.jnt_range[joint_id, 1]))
        self.qpos_adrs = np.array(self.qpos_adrs)
        self.robot_lo = np.array(self.robot_lo)
        self.robot_hi = np.array(self.robot_hi)

        self.shoulder_id = MJ_NAME2ID(model, MJT_OBJ.mjOBJ_BODY, self.shoulder_body)
        self.elbow_id = MJ_NAME2ID(model, MJT_OBJ.mjOBJ_BODY, self.elbow_body)
        self.wrist_id = MJ_NAME2ID(model, MJT_OBJ.mjOBJ_BODY, self.wrist_body)
        self.tcp_id = MJ_NAME2ID(model, MJT_OBJ.mjOBJ_BODY, self.tcp_body)

        arm_dim = len(self.arm_joint_names)
        self.eff_lo = np.empty(arm_dim)
        self.eff_hi = np.empty(arm_dim)
        for index in range(min(6, arm_dim)):
            human_lo, human_hi = HUMAN_LIMITS_RAD[index]
            self.eff_lo[index] = max(human_lo, self.robot_lo[index])
            self.eff_hi[index] = min(human_hi, self.robot_hi[index])
        if arm_dim > 6:
            self.eff_lo[6] = self.robot_lo[6]
            self.eff_hi[6] = self.robot_hi[6]

        self._prev_q = None
        self._prev_ref = None

    def _soechting_elbow(self, shoulder_pos, wrist_pos):
        shoulder_to_wrist_muj = wrist_pos - shoulder_pos
        shoulder_to_wrist_med = _muj2med(shoulder_to_wrist_muj)
        radius_actual = np.linalg.norm(shoulder_to_wrist_med)
        if radius_actual < 1e-8:
            return None
        radius_cm = radius_actual * (60.0 / L_TOTAL)
        phi = np.arctan2(shoulder_to_wrist_med[1], -shoulder_to_wrist_med[2]) * 180.0 / np.pi
        chi = np.arcsin(np.clip(shoulder_to_wrist_med[0] / radius_actual, -1, 1)) * 180.0 / np.pi
        q_eu_deg, q_yu_deg = self.map_fn(radius_cm, phi, chi)
        q_eu = np.deg2rad(q_eu_deg)
        q_yu = np.deg2rad(q_yu_deg)
        lu_med = np.array([
            np.sin(q_eu) * np.sin(q_yu),
            np.sin(q_eu) * np.cos(q_yu),
            -np.cos(q_eu),
        ])
        return shoulder_pos + L_UPPER * _med2muj(lu_med)

    @staticmethod
    def _ori_error(target_rotation, current_rotation):
        rot_error = target_rotation @ current_rotation.T
        trace = np.clip(rot_error.trace(), -1.0, 3.0)
        angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        if angle < 1e-8:
            return np.zeros(3)
        axis = np.array([
            rot_error[2, 1] - rot_error[1, 2],
            rot_error[0, 2] - rot_error[2, 0],
            rot_error[1, 0] - rot_error[0, 1],
        ])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            return np.zeros(3)
        return (angle / axis_norm) * axis

    def _compute_ori_scale(self, blend):
        blend_value = float(np.clip(blend, 0.0, 1.0))
        blend_value = 3.0 * blend_value * blend_value - 2.0 * blend_value * blend_value * blend_value
        return self.ori_constraint_max * blend_value

    def project(self, sim_or_model, qpos_in=None, return_debug=True, approach_blend=0.0):
        if isinstance(sim_or_model, tuple):
            model, data = sim_or_model
        else:
            model, data = sim_or_model.model, sim_or_model.data
        if hasattr(model, "_model"):
            model = model._model
        if hasattr(data, "_data"):
            data = data._data

        arm_dim = len(self.arm_joint_names)
        ori_scale = self._compute_ori_scale(approach_blend)

        full_qpos = data.qpos.copy()
        if qpos_in is not None:
            full_qpos[self.qpos_adrs] = qpos_in
        data.qpos[:] = full_qpos
        MJ_FWD_POSITION(model, data)

        tcp_target = data.xpos[self.tcp_id].copy()
        tcp_rot_demo = data.xmat[self.tcp_id].reshape(3, 3).copy()
        q_ref = full_qpos[self.qpos_adrs].copy()

        if self._prev_q is not None and self._prev_ref is not None:
            q0 = self._prev_q + (q_ref - self._prev_ref)
            q0 = np.clip(q0, self.robot_lo, self.robot_hi)
        else:
            q0 = q_ref.copy()

        def _fk(arm_qpos):
            full_qpos[self.qpos_adrs] = arm_qpos
            data.qpos[:] = full_qpos
            MJ_FWD_POSITION(model, data)
            return (
                data.xpos[self.shoulder_id].copy(),
                data.xpos[self.elbow_id].copy(),
                data.xpos[self.wrist_id].copy(),
                data.xpos[self.tcp_id].copy(),
                data.xmat[self.tcp_id].reshape(3, 3).copy(),
            )

        def _objective(arm_qpos):
            shoulder_pos, elbow_pos, wrist_pos, _tcp_pos, _tcp_rot = _fk(arm_qpos)
            cost = 0.0
            elbow_target = self._soechting_elbow(shoulder_pos, wrist_pos)
            if elbow_target is not None:
                cost += self.w_elbow * np.sum((elbow_pos - elbow_target) ** 2)
            for index in range(min(6, arm_dim)):
                lo, hi = self.eff_lo[index], self.eff_hi[index]
                if arm_qpos[index] < lo:
                    cost += self.w_limit * (lo - arm_qpos[index]) ** 2
                elif arm_qpos[index] > hi:
                    cost += self.w_limit * (arm_qpos[index] - hi) ** 2
            if self.w_temporal > 0 and self._prev_q is not None:
                cost += self.w_temporal * np.sum((arm_qpos - self._prev_q) ** 2)
            cost += self.w_posture * np.sum((arm_qpos - q_ref) ** 2)
            return cost

        def _eef_pos_constraint(arm_qpos):
            return _fk(arm_qpos)[3] - tcp_target

        constraints = [{"type": "eq", "fun": _eef_pos_constraint}]
        if ori_scale > 1e-4:
            def _eef_ori_constraint(arm_qpos):
                rotation = _fk(arm_qpos)[4]
                return ori_scale * self._ori_error(tcp_rot_demo, rotation)

            constraints.append({"type": "eq", "fun": _eef_ori_constraint})

        bounds = list(zip(self.robot_lo, self.robot_hi))
        result = scipy_minimize(
            _objective,
            q0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.slsqp_maxiter, "ftol": self.slsqp_ftol},
        )

        arm_qpos = result.x.copy()
        shoulder_pos, elbow_pos, wrist_pos, tcp_pos, tcp_rot = _fk(arm_qpos)
        tcp_err = float(np.linalg.norm(tcp_target - tcp_pos))
        ori_err = float(np.linalg.norm(self._ori_error(tcp_rot_demo, tcp_rot)) * 180.0 / np.pi)

        full_qpos[self.qpos_adrs] = arm_qpos
        data.qpos[:] = full_qpos
        MJ_FWD_POSITION(model, data)

        self._prev_q = arm_qpos.copy()
        self._prev_ref = q_ref.copy()

        debug = {}
        if return_debug:
            elbow_target = self._soechting_elbow(shoulder_pos, wrist_pos)
            debug["tcp_err"] = tcp_err
            debug["ori_err_deg"] = ori_err
            debug["approach_blend"] = float(approach_blend)
            debug["elbow_err"] = float(np.linalg.norm(elbow_target - elbow_pos)) if elbow_target is not None else 0.0

        return full_qpos, debug


def _load_npz_init_pose(npz_root: Path, task_stem: str, demo_idx: int) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Try humanized_sim.npz first, then fall back to humanized.npz."""
    demo_dir = npz_root / f"{task_stem}_demo" / "humanized_demo" / f"demo_{demo_idx:02d}"
    for filename in ("humanized_sim.npz", "humanized.npz"):
        npz_path = demo_dir / filename
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        q = np.array(data["joint_states_human"][0], dtype=np.float32)[:7]
        return q, filename
    return None, None


def _expand_suite_names(suite_names: Sequence[str]) -> list[str]:
    expanded = []
    for suite_name in suite_names:
        expanded.extend(SUITE_NAME_ALIASES.get(suite_name, [suite_name]))

    ordered_unique = []
    seen = set()
    for suite_name in expanded:
        if suite_name in seen:
            continue
        seen.add(suite_name)
        ordered_unique.append(suite_name)
    return ordered_unique


def _expand_npz_roots(npz_roots: Optional[Sequence[str]], suite_names: Sequence[str]) -> dict[str, Path]:
    if not npz_roots:
        return {}

    if len(npz_roots) == 1:
        root_template = npz_roots[0]
        if "___" in root_template:
            roots_by_suite = {}
            for suite_name in suite_names:
                token = SUITE_ROOT_TOKENS.get(suite_name)
                if token is None:
                    raise ValueError(f"Suite {suite_name} does not support ___ npz_root expansion")
                roots_by_suite[suite_name] = Path(root_template.replace("___", f"_{token}_"))
            return roots_by_suite
        if len(suite_names) == 1:
            return {suite_names[0]: Path(root_template)}
        raise ValueError("Multiple suites require one --npz_root per suite, or one template path containing ___")

    if len(npz_roots) != len(suite_names):
        raise ValueError("Number of --npz_root values must match the expanded --suite_name list")

    return {suite_name: Path(root) for suite_name, root in zip(suite_names, npz_roots)}


def _load_existing_results(output_path: Path) -> dict[str, dict[str, list[list[float]]]]:
    if not output_path.exists():
        return {}

    spec = importlib.util.spec_from_file_location("humanized_initial_pose_store", output_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import existing pose store from {output_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    existing = getattr(module, "HUMANIZED_INITIAL_ARM_JOINTS_BY_SUITE", {})
    return {
        suite_name: {
            task_stem: [list(map(float, pose)) for pose in poses]
            for task_stem, poses in tasks.items()
        }
        for suite_name, tasks in existing.items()
    }


def _sort_suite_names(suite_names: Sequence[str]) -> list[str]:
    order = {suite_name: index for index, suite_name in enumerate(DEFAULT_SUITE_NAMES)}
    return sorted(suite_names, key=lambda suite_name: (order.get(suite_name, len(order)), suite_name))


def generate_humanized_initial_poses(suite_name: str, npz_root: Optional[Path] = None):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    results = {}

    used_humanized_npz = []
    projector_fallback_tasks = []

    for task_id in range(len(suite.tasks)):
        task = suite.get_task(task_id)
        task_stem = Path(task.bddl_file).stem
        initial_states = suite.get_task_init_states(task_id)

        # When npz_root is provided, try loading directly from humanized_sim.npz first,
        # then fall back to humanized.npz.
        if npz_root is not None:
            task_poses = []
            needs_projector = []
            for demo_idx in range(len(initial_states)):
                q, source_filename = _load_npz_init_pose(npz_root, task_stem, demo_idx)
                task_poses.append(q)  # None = needs projector fallback
                if source_filename == "humanized.npz":
                    used_humanized_npz.append(f"{task_stem}/demo_{demo_idx:02d}")
                if q is None:
                    needs_projector.append(demo_idx)
            if not needs_projector:
                # All demos loaded from npz; no env needed.
                results[task_stem] = task_poses
                loaded = len(task_poses)
                print(f"  {task_stem}: loaded {loaded}/{loaded} from npz")
                continue
            # Some demos missing from npz; fall back to projector for those.
            projector_fallback_tasks.append(f"{task_stem} ({len(needs_projector)} demos)")
            print(f"  {task_stem}: {len(needs_projector)} demos missing from npz, using projector fallback")
        else:
            needs_projector = list(range(len(initial_states)))
            task_poses = [None] * len(initial_states)

        env, _ = get_libero_env(task, "openvla", resolution=256, use_joint_pos=True, joint_substeps=1)
        env.reset()

        projector = HumanElbowProjector(
            arm_joint_names=[f"robot0_joint{i}" for i in range(1, 8)],
            shoulder_body="robot0_link2",
            elbow_body="robot0_link4",
            wrist_body="robot0_link6",
            w_temporal=30.0,
            w_ori=0.0,
            approach_ori_boost=14.0,
            approach_elbow_damp=0.95,
            w_elbow=36.0,
            w_link7=6.0,
            w_posture=0.05,
        )
        projector.bind(env.sim)

        for demo_idx in needs_projector:
            obs = env.set_init_state(initial_states[demo_idx])
            original_q = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
            projector._prev_q = None
            projector._prev_ref = None
            q_full, _ = projector.project(env.sim, qpos_in=original_q, return_debug=True, approach_blend=0.0)
            task_poses[demo_idx] = np.asarray(q_full[projector.qpos_adrs], dtype=np.float32)

        results[task_stem] = task_poses
        try:
            env.close()
        except Exception:
            pass

    if npz_root is not None and used_humanized_npz:
        preview = ", ".join(used_humanized_npz[:5])
        extra = "" if len(used_humanized_npz) <= 5 else f", ... (+{len(used_humanized_npz) - 5} more)"
        print(
            f"WARNING [{suite_name}] {npz_root}: humanized_sim.npz missing for {len(used_humanized_npz)} demos; "
            f"fell back to humanized.npz for {preview}{extra}"
        )

    if npz_root is not None and projector_fallback_tasks:
        preview = ", ".join(projector_fallback_tasks[:5])
        extra = "" if len(projector_fallback_tasks) <= 5 else f", ... (+{len(projector_fallback_tasks) - 5} more tasks)"
        print(
            f"WARNING [{suite_name}] {npz_root}: projector fallback used for {len(projector_fallback_tasks)} tasks: "
            f"{preview}{extra}"
        )

    return results


def generate_humanized_initial_poses_for_suites(suite_names, npz_roots_by_suite: Optional[dict[str, Path]] = None):
    npz_roots_by_suite = npz_roots_by_suite or {}
    return {
        suite_name: generate_humanized_initial_poses(suite_name, npz_root=npz_roots_by_suite.get(suite_name))
        for suite_name in suite_names
    }


def write_output(results_by_suite, output_path: Path, suite_names):
    lines = []
    suite_names = _sort_suite_names(suite_names)
    suite_list = ", ".join(suite_names)
    lines.append(f'"""Stored humanized initial arm joint poses for suites: {suite_list}.')
    lines.append("")
    lines.append("Generated from benchmark initial simulator states using the copied")
    lines.append("HumanElbowProjector logic in A_generate_humanized_initial_poses.py.")
    lines.append("When --npz_root is supplied, joint_states_human[0] is read directly")
    lines.append("from pre-computed humanized_sim.npz files, falling back to humanized.npz.")
    lines.append('"""')
    lines.append("")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("HUMANIZED_INITIAL_ARM_JOINTS_BY_SUITE = {")
    for suite_name in suite_names:
        results = results_by_suite[suite_name]
        lines.append(f'    "{suite_name}": {{')
        for task_stem, poses in results.items():
            lines.append(f'        "{task_stem}": [')
            for pose in poses:
                pose_array = np.asarray(pose, dtype=np.float32)
                pose_str = ", ".join(f"{float(value):.8f}" for value in pose_array.tolist())
                lines.append(f"            [{pose_str}],")
            lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def get_humanized_initial_arm_joint_pose(suite_name: str, task_stem: str, episode_idx: int):")
    lines.append("    suite_poses = HUMANIZED_INITIAL_ARM_JOINTS_BY_SUITE.get(suite_name)")
    lines.append("    if suite_poses is None:")
    lines.append("        return None")
    lines.append("    poses = suite_poses.get(task_stem)")
    lines.append("    if poses is None or episode_idx < 0 or episode_idx >= len(poses):")
    lines.append("        return None")
    lines.append("    return np.asarray(poses[episode_idx], dtype=np.float32)")
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_name", nargs="*", default=DEFAULT_SUITE_NAMES)
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--npz_root",
        nargs="*",
        default=None,
        help="One root directory per suite, or one template path containing ___. "
             "Examples: /path/.../libero_10_humanized or /path/.../libero___humanized_vbasep21. "
             "When provided, joint_states_human[0] is read directly from "
             "{npz_root}/{task}_demo/humanized_demo/demo_{i:02d}/humanized_sim.npz first, "
             "then humanized.npz, instead of running the projector.",
    )
    args = parser.parse_args()

    suite_names = _expand_suite_names(args.suite_name or DEFAULT_SUITE_NAMES)
    output_path = Path(args.output_path)
    npz_roots_by_suite = _expand_npz_roots(args.npz_root, suite_names)
    updated_results = generate_humanized_initial_poses_for_suites(suite_names, npz_roots_by_suite=npz_roots_by_suite)

    merged_results = _load_existing_results(output_path)
    merged_results.update(updated_results)

    write_output(merged_results, output_path, merged_results.keys())
    total_tasks = sum(len(tasks) for tasks in updated_results.values())
    total_states = sum(len(poses) for tasks in updated_results.values() for poses in tasks.values())
    print(
        f"Updated {output_path} for suites {', '.join(suite_names)} "
        f"with {total_tasks} tasks and {total_states} initial states"
    )


if __name__ == "__main__":
    main()