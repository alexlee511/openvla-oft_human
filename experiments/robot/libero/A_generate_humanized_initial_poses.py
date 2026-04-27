"""Generate stored humanized initial arm poses from LIBERO benchmark initial states."""

from __future__ import annotations

import argparse
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
        w_elbow: float = 1.0,
        w_limit: float = 1.0,
        w_posture: float = 0.001,
        w_temporal: float = 0.1,
        ori_constraint_max: float = 1.0,
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


def generate_humanized_initial_poses(suite_name: str):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    results = {}

    for task_id in range(len(suite.tasks)):
        task = suite.get_task(task_id)
        task_stem = Path(task.bddl_file).stem
        initial_states = suite.get_task_init_states(task_id)
        env, _ = get_libero_env(task, "openvla", resolution=256, use_joint_pos=True, joint_substeps=1)
        env.reset()

        projector = HumanElbowProjector(
            arm_joint_names=[f"robot0_joint{i}" for i in range(1, 8)],
            shoulder_body="robot0_link2",
            elbow_body="robot0_link4",
            wrist_body="robot0_link6",
        )
        projector.bind(env.sim)

        task_poses = []
        for initial_state in initial_states:
            obs = env.set_init_state(initial_state)
            original_q = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
            projector._prev_q = None
            projector._prev_ref = None
            q_full, _ = projector.project(env.sim, qpos_in=original_q, return_debug=True, approach_blend=0.0)
            task_poses.append(np.asarray(q_full[projector.qpos_adrs], dtype=np.float32))

        results[task_stem] = task_poses
        try:
            env.close()
        except Exception:
            pass

    return results


def generate_humanized_initial_poses_for_suites(suite_names):
    return {suite_name: generate_humanized_initial_poses(suite_name) for suite_name in suite_names}


def write_output(results_by_suite, output_path: Path, suite_names):
    lines = []
    suite_list = ", ".join(suite_names)
    lines.append(f'"""Stored humanized initial arm joint poses for suites: {suite_list}.')
    lines.append("")
    lines.append("Generated from benchmark initial simulator states using the copied")
    lines.append("HumanElbowProjector logic in A_generate_humanized_initial_poses.py")
    lines.append("with approach_blend=0.0, then stored for eval-time reuse.")
    lines.append('"""')
    lines.append("")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("HUMANIZED_INITIAL_ARM_JOINTS_BY_SUITE = {")
    for suite_name, results in results_by_suite.items():
        lines.append(f'    "{suite_name}": {{')
        for task_stem, poses in results.items():
            lines.append(f'        "{task_stem}": [')
            for pose in poses:
                pose_str = ", ".join(f"{float(value):.8f}" for value in pose.tolist())
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
    args = parser.parse_args()

    suite_names = args.suite_name or DEFAULT_SUITE_NAMES
    results = generate_humanized_initial_poses_for_suites(suite_names)
    output_path = Path(args.output_path)
    write_output(results, output_path, suite_names)
    total_tasks = sum(len(tasks) for tasks in results.values())
    total_states = sum(len(poses) for tasks in results.values() for poses in tasks.values())
    print(f"Wrote {output_path} with {len(suite_names)} suites, {total_tasks} tasks, and {total_states} initial states")


if __name__ == "__main__":
    main()