2. Absolute → Delta Conversion for Eval
This is the key modification needed for eval. The current run_libero_eval.py uses OSC_POSE (7D delta EEF). For joint position control, you need:

What needs to change in eval:
(a) Environment creation — pass controller="JOINT_POSITION" to ControlEnv:

The ControlEnv.__init__ in env_wrapper.py:17 already supports this kwarg — just defaults to "OSC_POSE". So modify get_libero_env() to pass it.

(b) Absolute → delta conversion — the model outputs absolute joint positions, but JOINT_POSITION controller expects normalized delta action[:7] = dq / dq_max ∈ [-1, 1]. This is exactly what joints_target_to_action() does at A_libero_joint_replay.py:431:

Edited
transforms.py+2-3
At eval you need the same logic, but pulling q_cur from the sim observation:

(c) Skip process_action() — the existing gripper processing (normalize_gripper_action + invert_gripper_action) is for the OSC_POSE pipeline where gripper was stored as [0,1] inverted. Our gripper is already raw -1/+1, so no processing needed.

(d) Change prepare_observation() — proprioceptive state must be [joint_pos(7), gripper_width(1)] instead of [eef_pos(3), axisangle(3), gripper_qpos(2)]:

(e) Change get_libero_dummy_action() — 8D: [0,0,0,0,0,0,0, -1] (7 zero deltas + open gripper)

(f) num_open_loop_steps — should match NUM_ACTIONS_CHUNK = 25

dq_max at eval time
The tricky part is getting ctrl.output_max. You have two options:

Access the controller via env.env.robots[0].controller.output_max after env.reset()
Hardcode the Panda JOINT_POSITION default (typically 0.05 per joint per step at 20Hz control)
