import mujoco
import mujoco_viewer
import numpy as np
import yaml
import torch
import os

# 🌟 真正匹配你 XML 的关节名称 (严格对齐 IsaacGym 默认排布顺序) 🌟
ACTUATED_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_joint", "head_joint",
    "left_arm_pitch_joint", "left_arm_roll_joint", "left_arm_yaw_joint", "left_elbow_joint",
    "right_arm_pitch_joint", "right_arm_roll_joint", "right_arm_yaw_joint", "right_elbow_joint"
]

KEY_BODY_NAMES = [
    "head_link",
    "right_hand_link",
    "left_hand_link",
    "right_ankle_roll_link",
    "left_ankle_roll_link"
]


# ================= 工具函数 =================
def quat_from_angle_axis(angle, axis):
    sin_half = np.sin(angle / 2.0)
    cos_half = np.cos(angle / 2.0)
    norm = np.linalg.norm(axis)
    axis_normalized = axis / np.maximum(norm, 1e-8)
    xyz = axis_normalized * sin_half
    w = cos_half
    return np.array([xyz[0], xyz[1], xyz[2], w])


def quat_rotate(q, v):
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c


def quat_to_tan_norm(q):
    ref_tan = np.array([1.0, 0.0, 0.0])
    tan = quat_rotate(q, ref_tan)
    ref_norm = np.array([0.0, 0.0, 1.0])
    norm = quat_rotate(q, ref_norm)
    return np.concatenate([tan, norm])


# ================= 核心流转 =================
def get_mujoco_data(data, model):
    mujoco_data = {}
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)

    quat_xyzw = np.array([q[4], q[5], q[6], q[3]])

    mujoco_data['mujoco_root_pos'] = q[:3]
    mujoco_data['mujoco_root_rot'] = quat_xyzw
    mujoco_data['mujoco_root_vel'] = dq[:3]

    local_ang_vel = dq[3:6]
    mujoco_data['mujoco_root_angVel'] = quat_rotate(quat_xyzw, local_ang_vel)

    # 严格按照名字提取关节数据
    dof_pos_list = []
    dof_vel_list = []
    for jnt_name in ACTUATED_JOINT_NAMES:
        jnt_id = model.joint(jnt_name).id
        qpos_adr = model.jnt_qposadr[jnt_id]
        qvel_adr = model.jnt_dofadr[jnt_id]
        dof_pos_list.append(data.qpos[qpos_adr])
        dof_vel_list.append(data.qvel[qvel_adr])

    mujoco_data['mujoco_dof_pos'] = np.array(dof_pos_list)
    mujoco_data['mujoco_dof_vel'] = np.array(dof_vel_list)

    key_pos = []
    for body_name in KEY_BODY_NAMES:
        body_id = model.body(name=body_name).id
        key_pos.append(data.xpos[body_id])
    mujoco_data['mujoco_key_pos'] = np.array(key_pos)

    joint_quats = []
    for i in range(2, model.nbody):
        jnt_num = model.body_jntnum[i]
        if jnt_num == 1:
            jnt_adr = model.body_jntadr[i]
            angle = data.qpos[model.jnt_qposadr[jnt_adr]]
            axis = model.jnt_axis[jnt_adr]
            q_local = quat_from_angle_axis(angle, axis)
        else:
            q_local = np.array([0.0, 0.0, 0.0, 1.0])
        joint_quats.append(q_local)

    mujoco_data['mujoco_joint_quats'] = np.array(joint_quats)
    return mujoco_data


def compute_humanoid_observations(mujoco_data):
    root_h = mujoco_data['mujoco_root_pos'][2:3]
    root_rot_obs = quat_to_tan_norm(mujoco_data['mujoco_root_rot'])
    root_vel_obs = mujoco_data['mujoco_root_vel']
    root_ang_vel_obs = mujoco_data['mujoco_root_angVel']
    joint_quats = mujoco_data['mujoco_joint_quats']
    joint_rot_obs = np.array([quat_to_tan_norm(q) for q in joint_quats]).flatten()
    dof_vel = mujoco_data['mujoco_dof_vel']
    root_pos = mujoco_data['mujoco_root_pos']
    key_body_pos = mujoco_data['mujoco_key_pos'] - root_pos
    key_pos_flat = key_body_pos.flatten()

    obs = np.concatenate([
        root_h, root_rot_obs, root_vel_obs, root_ang_vel_obs,
        joint_rot_obs, dof_vel, key_pos_flat
    ])
    return obs.astype(np.float32).reshape(1, -1)


# ================= 主运行逻辑 =================
def run_mujoco():
    # 1. 动态获取根目录 (无论你在哪里执行该脚本，都能精准找到 MimicKit 根目录)
    # 当前脚本位于 MimicKit/deploy/ig_deploy/main_ig.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))

    # env_yaml_path = os.path.join(project_root, "data/envs/amp_v47_cmu_env_base.yaml")
    # engine_yaml_path = os.path.join(project_root, "data/engines/isaac_gym_engine.yaml")
    # xml_path = os.path.join(project_root, "data/assets/v47/mjcf/v47_inertia_v3all_base.xml")
    # jit_path = os.path.join(project_root, "output/amp_v47_cmu_env_base/2026-0302_105149/model_jit.pt")

    env_yaml_path = os.path.join(project_root, "data/envs/amp_v47_cmu_env.yaml")
    engine_yaml_path = os.path.join(project_root, "data/engines/isaac_gym_engine.yaml")
    xml_path = os.path.join(project_root, "data/assets/v47/mjcf/v47_inertia_v3all_mujoco.xml")
    jit_path = os.path.join(project_root, "output/amp_v47_cmu_env/2026-0228_152123/model_jit.pt")
    #
    with open(env_yaml_path, "r") as f:
        env_cfg = yaml.safe_load(f)

    control_freq = 30
    simulation_dt = 0.001
    decimation = int((1.0 / control_freq) / simulation_dt)

    init_pose_list = env_cfg["init_pose"]
    default_dof_pos_arr = np.array(init_pose_list[1:])

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt

    # 对齐新的真实关节名称
    kps_mapping = {
        "hip_pitch": 200, "hip_roll": 150, "hip_yaw": 100, "knee": 400, "ankle_pitch": 40, "ankle_roll": 30,
        "waist": 100, "head": 20, "arm_pitch": 40, "arm_roll": 40, "arm_yaw": 20, "elbow": 20
    }
    kds_mapping = {
        "hip_pitch": 10, "hip_roll": 5, "hip_yaw": 5, "knee": 10, "ankle_pitch": 2.5, "ankle_roll": 2,
        "waist": 5, "head": 1, "arm_pitch": 2, "arm_roll": 2, "arm_yaw": 1, "elbow": 1
    }

    mapped_kps_arr = np.zeros(22)
    mapped_kds_arr = np.zeros(22)
    actuator_indices = []

    for i, jnt_name in enumerate(ACTUATED_JOINT_NAMES):
        for key in kps_mapping.keys():
            if key in jnt_name:
                mapped_kps_arr[i] = kps_mapping[key]
                mapped_kds_arr[i] = kds_mapping[key]
                break

        # 🌟 终极防错位：获取该关节在 XML 中对应的真实 Actuator ID 🌟
        jnt_id = model.joint(jnt_name).id
        act_id = np.where(model.actuator_trnid[:, 0] == jnt_id)[0][0]
        actuator_indices.append(act_id)

    gears = np.copy(model.actuator_gear[:, 0])

    for i, jnt_name in enumerate(ACTUATED_JOINT_NAMES):
        qpos_adr = model.jnt_qposadr[model.joint(jnt_name).id]
        data.qpos[qpos_adr] = default_dof_pos_arr[i]

    mujoco.mj_step(model, data)

    viewer = mujoco_viewer.MujocoViewer(model, data)
    model.vis.map.rgba_force[3] = 0.5
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -45
    viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

    print(f"Loading JIT Policy from: {jit_path}")
    policy = torch.jit.load(jit_path)
    policy.eval()

    count = 0
    target_dof_pos = default_dof_pos_arr.copy()
    target_dof_vel = np.zeros(22, dtype=np.double)

    print("Running...")
    while viewer.is_alive:
        mujoco_data = get_mujoco_data(data, model)
        policy_active = True

        if policy_active and count % decimation == 0:
            obs_buff = compute_humanoid_observations(mujoco_data)
            obs_tensor = torch.tensor(obs_buff, dtype=torch.float32)

            with torch.no_grad():
                action_tensor = policy(obs_tensor)

            net_output_pos = action_tensor.numpy()[0]

            articulated_range = []
            for jnt_name in ACTUATED_JOINT_NAMES:
                articulated_range.append(model.jnt_range[model.joint(jnt_name).id])
            articulated_range_arr = np.array(articulated_range)

            target_dof_pos = np.clip(net_output_pos, articulated_range_arr[:, 0], articulated_range_arr[:, 1])

        current_pos = mujoco_data["mujoco_dof_pos"]
        current_vel = mujoco_data["mujoco_dof_vel"]

        tau_pd = (target_dof_pos - current_pos) * mapped_kps_arr + (target_dof_vel - current_vel) * mapped_kds_arr

        # 🌟 根据匹配好的 Actuator ID 精准下发力矩，绝不依赖数组顺序 🌟
        for i, act_id in enumerate(actuator_indices):
            data.ctrl[act_id] = tau_pd[i] / gears[act_id]

        mujoco.mj_step(model, data)
        viewer.render()
        count += 1

    viewer.close()


if __name__ == "__main__":
    run_mujoco()