import enum
import gymnasium.spaces as spaces
import numpy as np
import os
import torch

import anim.motion_lib as motion_lib
import engines.engine as engine
import envs.sim_env as sim_env
import envs.base_env as base_env
import util.camera as camera

from util.logger import Logger
import util.torch_util as torch_util

import engines.engine as engine

class CharEnv(sim_env.SimEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._global_obs = env_config["global_obs"]
        self._root_height_obs = env_config.get("root_height_obs", True)
        self._zero_center_action = env_config.get("zero_center_action", False)

        self._enable_action_pd_limit = env_config.get("enable_action_pd_limit", True)  # 默认 True 兼容你以前的配置
        if self._enable_action_pd_limit:
            # 解析游隙阈值 (支持单浮点数，也支持逐关节的列表)
            pd_thresh = env_config.get("action_pd_threshold", 0.2)
            if isinstance(pd_thresh, list):
                self._action_pd_threshold = torch.tensor(pd_thresh, device=device, dtype=torch.float32)
            else:
                self._action_pd_threshold = float(pd_thresh)
        else:
            self._action_pd_threshold = None
        self._use_physical_bound_loss = env_config.get("use_physical_bound_loss", False)
        self._fix_root = env_config.get("fix_root", False)

        # ====== 新增：关节屏蔽配置 ======
        self._enable_joint_mask = env_config.get("enable_joint_mask", False)
        self._masked_joint_names = env_config.get("masked_joint_names", [])

        # 新增：是否连带屏蔽掉 hand、toe 等假关节的开关，方便你做消融实验对比
        self._mask_fake_joints = env_config.get("mask_fake_joints", False)
        # =================================

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        
        char_id = self._get_char_id()
        self._print_char_prop(0, char_id)
        self._validate_envs()
        return

    def _parse_init_pose(self, init_pose, device):
        dof_size = self._kin_char_model.get_dof_size()

        if (init_pose is None):
            init_pose = torch.zeros(6 + dof_size, dtype=torch.float32, device=device)
        else:
            init_pose = torch.tensor(init_pose, device=device)

            if (init_pose.shape[-1] == 3):
                pad_pose = torch.zeros(3 + dof_size, dtype=torch.float32, device=device)
                init_pose = torch.cat([init_pose, pad_pose], dim=-1)
            
        init_root_pos, init_root_rot, init_dof_pos = motion_lib.extract_pose_data(init_pose)
        assert(init_dof_pos.shape[-1] == dof_size)

        self._init_root_pos = init_root_pos
        self._init_root_rot = torch_util.exp_map_to_quat(init_root_rot)
        self._init_dof_pos = init_dof_pos
        return

    def _parse_init_pose_real(self, init_pose_real, device):
        dof_size = self._kin_char_model.get_dof_size()

        if (init_pose_real is None):
            init_pose_real = torch.zeros(6 + dof_size, dtype=torch.float32, device=device)
        else:
            init_pose_real = torch.tensor(init_pose_real, device=device)

            if (init_pose_real.shape[-1] == 3):
                pad_pose = torch.zeros(3 + dof_size, dtype=torch.float32, device=device)
                init_pose_real = torch.cat([init_pose_real, pad_pose], dim=-1)

        init_root_pos_real, init_root_rot_real, init_dof_pos_real = motion_lib.extract_pose_data(init_pose_real)
        assert(init_dof_pos_real.shape[-1] == dof_size)

        self._init_root_pos_real = init_root_pos_real
        self._init_root_rot_real = torch_util.exp_map_to_quat(init_root_rot_real)
        self._init_dof_pos_real = init_dof_pos_real
        return

    def _build_envs(self, env_config, num_envs):
        char_file = env_config["char_file"]
        self._build_kin_char_model(char_file)
        
        init_pose = env_config.get("init_pose", None)
        self._parse_init_pose(init_pose, self._device)

        # ====== 新增：解析真实的初始位姿 ======
        init_pose_real = env_config.get("init_pose_real", None)
        self._parse_init_pose_real(init_pose_real, self._device)

        self._char_ids = []
        
        for e in range(num_envs):
            Logger.print("Building {:d}/{:d} envs".format(e + 1, num_envs), end='\r')
            env_id = self._engine.create_env()
            assert(env_id == e)
            self._build_env(env_id, env_config)

        Logger.print("\n")
        return
    
    def _build_env(self, env_id, env_config):
        char_col = self._get_char_color()
        char_id = self._build_character(env_id, env_config, color=char_col)

        if (env_id == 0):
            self._char_ids.append(char_id)
        else:
            char_id0 = self._char_ids[0]
            assert(char_id0 == char_id)
        
        return 
    
    def _build_character(self, env_id, env_config, color=None):
        char_file = env_config["char_file"]

        # 从 yaml init_pose 构建关节名→初始位置字典，用于满足 USD 关节限位校验
        init_joint_pos = {}
        dof_idx = 0
        num_joints = self._kin_char_model.get_num_joints()
        for j in range(1, num_joints):
            joint = self._kin_char_model.get_joint(j)
            dof_dim = joint.get_dof_dim()
            if dof_dim == 1:
                init_joint_pos[joint.name] = float(self._init_dof_pos[dof_idx].item())
                dof_idx += 1

        char_id = self._engine.create_obj(env_id=env_id,
                                          obj_type=engine.ObjType.articulated,
                                          asset_file=char_file,
                                          name="character",
                                          start_pos=self._init_root_pos.cpu().numpy(),
                                          start_rot=self._init_root_rot.cpu().numpy(),
                                          fix_root=self._fix_root,
                                          color=color,
                                          init_joint_pos=init_joint_pos)
        return char_id
    
    def _build_kin_char_model(self, char_file):
        _, file_ext = os.path.splitext(char_file)
        if (file_ext == ".xml"):
            import anim.mjcf_char_model as mjcf_char_model
            char_model = mjcf_char_model.MJCFCharModel(self._device)
        elif (file_ext == ".urdf"):
            import anim.urdf_char_model as urdf_char_model
            char_model = urdf_char_model.URDFCharModel(self._device)
        else:
            print("Unsupported character file format: {:s}".format(file_ext))
            assert(False)

        self._kin_char_model = char_model
        self._kin_char_model.load(char_file)
        return
    
    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)
        
        self._action_bound_low = torch.tensor(self._action_space.low, device=self._device)
        self._action_bound_high = torch.tensor(self._action_space.high, device=self._device)
        
        key_bodies = env_config.get("key_bodies", [])
        self._key_body_ids = self._build_body_ids_tensor(key_bodies)
        return
    
    def _build_action_space(self):
        control_mode = self._engine.get_control_mode()

        if (control_mode == engine.ControlMode.none):
            low, high = self._build_action_bounds_none()

        elif (control_mode == engine.ControlMode.vel):
            low, high = self._build_action_bounds_vel()

        elif (control_mode == engine.ControlMode.torque):
            char_id = self._get_char_id()
            torque_lim = self._engine.get_obj_torque_limits(0, char_id)
            low, high = self._build_action_bounds_torque(torque_lim)

        elif (control_mode == engine.ControlMode.pos
              or control_mode == engine.ControlMode.pd_explicit):       # <=
            char_id = self._get_char_id()
            dof_low, dof_high = self._engine.get_obj_dof_limits(0, char_id)
            low, high = self._build_action_bounds_pos(dof_low, dof_high)

        else:
            assert(False), "Unsupported control mode: {}".format(control_mode)
        
        # check to make sure that pd_explicit is only used for 1D joints
        if (control_mode == engine.ControlMode.pd_explicit):
            num_joints = self._kin_char_model.get_num_joints()
            for j in range(1, num_joints):
                j_dim = self._kin_char_model.get_joint_dof_dim(j)
                assert(j_dim <= 1), "pd_explicit only supports 1D joints"

        # ====== 核心修复：解耦 DOF 掩码与 Joint_Rot 掩码，并加入假关节开关 ======
        if getattr(self, "_enable_joint_mask", False) and len(self._masked_joint_names) > 0:
            char_id = self._get_char_id()
            dof_names = self._engine.get_obj_dof_names(char_id)

            masked_dof_indices = []
            masked_joint_rot_indices = []

            # 1. 匹配需要屏蔽的真实电机自由度 (处理 action, dof_pos, dof_vel)
            for name in self._masked_joint_names:
                if name in dof_names:
                    masked_dof_indices.append(dof_names.index(name))
            self._masked_dof_indices = torch.tensor(masked_dof_indices, dtype=torch.long, device=self._device)

            # 2. 匹配需要屏蔽的 joint_rot (处理观测)
            num_joints = self._kin_char_model.get_num_joints()
            for j in range(1, num_joints):
                joint = self._kin_char_model.get_joint(j)
                joint_name = joint.name

                # 情况 A: 它是被主动屏蔽的真实关节 (如 ankle, 上肢, 以及我们想焊死的 hip_roll)
                if joint_name in self._masked_joint_names:
                    masked_joint_rot_indices.append(j - 1)
                # 情况 B: 它是假关节 (没有自由度)，且你开启了假关节屏蔽实验
                elif getattr(self, "_mask_fake_joints", False) and joint.get_dof_dim() == 0:
                    masked_joint_rot_indices.append(j - 1)

            self._masked_joint_rot_indices = torch.tensor(masked_joint_rot_indices, dtype=torch.long,
                                                          device=self._device)

            # --- 缩减真实自由度对应的空间 ---
            dof_size = len(low)
            mask_dof = torch.ones(dof_size, dtype=torch.bool, device=self._device)
            if len(self._masked_dof_indices) > 0:
                mask_dof[self._masked_dof_indices] = False
            self._unmasked_dof_indices = torch.arange(dof_size, device=self._device)[mask_dof]

            # --- 缩减假关节包含的 joint_rot 空间 ---
            num_j_rot = num_joints - 1
            mask_j = torch.ones(num_j_rot, dtype=torch.bool, device=self._device)
            if len(self._masked_joint_rot_indices) > 0:
                mask_j[self._masked_joint_rot_indices] = False
            self._unmasked_joint_rot_indices = torch.arange(num_j_rot, device=self._device)[mask_j]

            # 缩减 low 和 high 数组
            low = low[mask_dof.cpu().numpy()]
            high = high[mask_dof.cpu().numpy()]

            # 缩减 PD 阈值
            if isinstance(self._action_pd_threshold, torch.Tensor) and len(self._masked_dof_indices) > 0:
                self._action_pd_threshold = self._action_pd_threshold[self._unmasked_dof_indices]
        else:
            self._unmasked_dof_indices = torch.arange(len(low), dtype=torch.long, device=self._device)
            self._unmasked_joint_rot_indices = torch.arange(self._kin_char_model.get_num_joints() - 1,
                                                            dtype=torch.long, device=self._device)
        # ========================================================

        action_space = spaces.Box(low=low, high=high)
        return action_space
    
    def _build_action_bounds_none(self):
        char_id = self._get_char_id()
        dof_pos = self._engine.get_dof_pos(char_id)
        action_size = int(dof_pos.shape[-1])
        low = -np.ones([action_size], dtype=np.float32)
        high = np.ones([action_size], dtype=np.float32)
        return low, high

    def _build_action_bounds_pos(self, dof_low, dof_high):
        low = np.zeros(dof_high.shape, dtype=np.float32)
        high = np.zeros(dof_high.shape, dtype=np.float32)

        num_joints = self._kin_char_model.get_num_joints()
        for j in range(1, num_joints):
            curr_joint = self._kin_char_model.get_joint(j)
            j_dof_dim = curr_joint.get_dof_dim()

            if (j_dof_dim > 0):
                if (j_dof_dim == 3): # 3D spherical j   # ×
                    # spherical joints are modeled as exponential maps
                    # so the bounds are computed a bit differently from revolute joints
                    j_low = curr_joint.get_joint_dof(dof_low)
                    j_high = curr_joint.get_joint_dof(dof_high)
                    j_low = np.max(np.abs(j_low))
                    j_high = np.max(np.abs(j_high))
                    curr_scale = max([j_low, j_high])
                    curr_scale = 1.2 * curr_scale

                    curr_low = -curr_scale
                    curr_high = curr_scale
                else:                               # <=
                    j_low = curr_joint.get_joint_dof(dof_low)
                    j_high = curr_joint.get_joint_dof(dof_high)

                    if (self._zero_center_action):  # <=
                        curr_mid = np.zeros_like(j_high)
                    else:
                        curr_mid = 0.5 * (j_high + j_low)

                    diff_high = np.abs(j_high - curr_mid)
                    diff_low = np.abs(j_low - curr_mid)
                    curr_scale = np.maximum(diff_high, diff_low)
                    shrink_ratio = 0.99
                    curr_scale *= shrink_ratio

                    curr_low = curr_mid - curr_scale
                    curr_high = curr_mid + curr_scale

                curr_joint.set_joint_dof(curr_low, low)
                curr_joint.set_joint_dof(curr_high, high)

        return low, high

    def _build_action_bounds_vel(self):
        char_id = self._get_char_id()
        dof_pos = self._engine.get_dof_pos(char_id)
        action_size = int(dof_pos.shape[-1])
        low = -2.0 * np.pi * np.ones([action_size], dtype=np.float32)
        high = 2.0 * np.pi * np.ones([action_size], dtype=np.float32)
        return low, high

    def _build_action_bounds_torque(self, torque_lim):
        char_id = self._get_char_id()
        dof_pos = self._engine.get_dof_pos(char_id)
        assert(dof_pos.shape[-1] == len(torque_lim))
        low = -np.array(torque_lim, dtype=np.float32)
        high = np.array(torque_lim, dtype=np.float32)
        return low, high
    
    def _print_char_prop(self, env_id, obj_id):
        num_dofs = self._engine.get_obj_num_dofs(obj_id)
        total_mass = self._engine.calc_obj_mass(env_id, obj_id)
        char_info = "Char {:d} properties\n\tDoFs: {:d}\n\tMass: {:.3f} kg\n".format(obj_id, num_dofs, total_mass)
        Logger.print(char_info)
        return
    
    def _validate_envs(self):
        # checks to make sure the kinematic model is consistent with the simulation model
        char_id = self._get_char_id()
        sim_body_names = self._engine.get_obj_body_names(char_id)
        kin_body_names = self._kin_char_model.get_body_names()

        for sim_name, kin_name in zip(sim_body_names, kin_body_names):
            assert(sim_name == kin_name)
        return
    
    def _get_char_id(self):
        return self._char_ids[0]
    
    def _update_reward(self):
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        self._reward_buf[:] = compute_reward(char_root_pos)
        return

    def _update_done(self):
        self._done_buf[:] = compute_done(self._done_buf, self._time_buf, 
                                         self._episode_length)
        return
    
    def _compute_obs(self, env_ids=None):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

        # ====== 新增：切除被屏蔽的关节数据 ======
        if getattr(self, "_enable_joint_mask", False) and len(self._masked_dof_indices) > 0:
            joint_rot_in = joint_rot[..., self._unmasked_joint_rot_indices, :]
            dof_vel_in = dof_vel[..., self._unmasked_dof_indices]
        else:
            joint_rot_in = joint_rot
            dof_vel_in = dof_vel
        # =====================================

        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        if hasattr(self, "commands"):
            commands = self.commands if env_ids is None else self.commands[env_ids]
        else:
            commands = torch.tensor([])
        # 此时已经拿到了 JIT 函数返回的纯净 tensor，且处于类方法中，可以安全使用 self
        if getattr(self, "_enable_push", False):
            # 只要 duration > 0，就认为是 1.0，否则是 0.0
            # 维度扩展为 [num_envs, 1]
            is_pushed_bool = (self._push_duration_buf > 0).float().unsqueeze(-1)
            # ====== 新增：修复维度炸裂的 BUG ======
            if env_ids is not None:
                is_pushed_bool = is_pushed_bool[env_ids]
            # ======================================
        else:
            is_pushed_bool = torch.tensor([])

        actor_obs, critic_obs, prop_obs, priv_obs = compute_char_obs(
                               root_pos=root_pos,
                               root_rot=root_rot, 
                               root_vel=root_vel,
                               root_ang_vel=root_ang_vel,
                               joint_rot=joint_rot_in,  # 修改此处
                               dof_vel=dof_vel_in,  # 修改此处
                               key_pos=key_pos,
                               global_obs=self._global_obs,
                               root_height_obs=self._root_height_obs,
                               cmds=commands,
                               push=is_pushed_bool,
                               asymmetric_obs=self._asymmetric_obs
                                                 )
        # 动态创建并保存 Critic 观测，避免破坏 sim_env.py 里的 self._obs_buf
        if not hasattr(self, '_critic_obs_buf'):
            self._critic_obs_buf = torch.zeros([self.get_num_envs(), critic_obs.shape[-1]], device=self._device)

        if env_ids is None:
            self._critic_obs_buf[:] = critic_obs
        else:
            self._critic_obs_buf[env_ids] = critic_obs

        # === 新增：将 critic_obs 放入 info 字典暴露给 Agent ===
        self._info["critic_obs"] = self._critic_obs_buf
        # ==========================================================
        # 【核心新增】：把物理引擎真实的 dof_pos 放进 info 字典，安全传递给 PPO！
        # 这里必须放缩减后的 dof_pos，否则 ppo_agent 计算 loss 维度不匹配！
        if getattr(self, "_enable_joint_mask", False) and len(self._masked_dof_indices) > 0:
            dof_pos_out = dof_pos[:, self._unmasked_dof_indices]
        else:
            dof_pos_out = dof_pos

        if not hasattr(self, '_dof_pos_buf'):
            self._dof_pos_buf = torch.zeros_like(dof_pos_out)
        if env_ids is None:
            self._dof_pos_buf[:] = dof_pos_out
        else:
            self._dof_pos_buf[env_ids] = dof_pos_out
        self._info["dof_pos"] = self._dof_pos_buf.clone()
        # ==========================================================
        # 只把 Actor 观测 return 给外层，这样 get_obs_space 自动推断维度就是 187
        return actor_obs

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_char(env_ids)
            self._reset_char_rigid_body_state(env_ids)
        return

    def _reset_char(self, env_ids):
        char_id = self._get_char_id()

        self._engine.set_root_pos(env_ids, char_id, self._init_root_pos)
        self._engine.set_root_rot(env_ids, char_id, self._init_root_rot)
        self._engine.set_root_vel(env_ids, char_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, char_id, 0.0)
        
        self._engine.set_dof_pos(env_ids, char_id, self._init_dof_pos)
        self._engine.set_dof_vel(env_ids, char_id, 0.0)
        
        self._engine.set_body_vel(env_ids, char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, char_id, 0.0)
        return

    def _reset_char_rigid_body_state(self, env_ids):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)[env_ids]
        root_rot = self._engine.get_root_rot(char_id)[env_ids]
        dof_pos = self._engine.get_dof_pos(char_id)[env_ids]

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        body_pos, body_rot = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)

        self._engine.set_body_pos(env_ids, char_id, body_pos)
        self._engine.set_body_rot(env_ids, char_id, body_rot)
        return

    def _apply_action(self, actions):
        char_id = self._get_char_id()

        curr_dof_pos = self._engine.get_dof_pos(char_id)

        # 1. 提取未屏蔽的当前关节角用于游隙限制
        if getattr(self, "_enable_joint_mask", False) and len(self._masked_dof_indices) > 0:
            curr_dof_pos_small = curr_dof_pos[:, self._unmasked_dof_indices]
        else:
            curr_dof_pos_small = curr_dof_pos

        if self._enable_action_pd_limit:

            # 适配 pd_threshold 张量的缩减
            if isinstance(self._action_pd_threshold, torch.Tensor) and getattr(self, "_enable_joint_mask", False):
                pd_thresh_small = self._action_pd_threshold[self._unmasked_dof_indices]
            else:
                pd_thresh_small = self._action_pd_threshold

            # 1. 先做相对裁剪（游隙控制/短狗绳）
            # 现在这里的 _action_pd_threshold 是一个张量，会自动对22个关节进行逐一宽容度计算！
            smooth_action = torch.clamp(
                actions,
                min=curr_dof_pos_small - pd_thresh_small,
                max=curr_dof_pos_small + pd_thresh_small
            )
        else:
            smooth_action = actions

        # 2. 再做绝对限位裁剪（物理兜底/公园围栏）
        # 不管网络怎么作妖，发给物理引擎的指令绝对不能超出 XML 设定的边界
        clip_action = torch.minimum(
            torch.maximum(smooth_action, self._action_bound_low),
            self._action_bound_high
        )

        # 2. 补全动作维度：让被屏蔽的关节锁定在初始设定的位姿参与仿真
        if getattr(self, "_enable_joint_mask", False) and len(self._masked_dof_indices) > 0:
            num_envs = actions.shape[0]
            full_actions = self._init_dof_pos.unsqueeze(0).expand(num_envs, -1).clone()
            full_actions[:, self._unmasked_dof_indices] = clip_action
            final_action = full_actions
        else:
            final_action = clip_action

        # 下发给底层引擎计算扭矩
        self._engine.set_cmd(char_id, final_action)
        return
    
    def _build_body_ids_tensor(self, body_names):
        char_id = self._get_char_id()
        body_ids = []

        for body_name in body_names:
            body_id = self._engine.find_obj_body_id(char_id, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = torch.tensor(body_ids, device=self._device, dtype=torch.long)
        return body_ids
    
    def _has_key_bodies(self):
        return len(self._key_body_ids) > 0

    def _build_camera(self, env_config):
        env_id = 0
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_pos = char_root_pos[env_id].cpu().numpy()
            
        cam_pos = np.array([char_pos[0], char_pos[1] - 5.0, 3.0])
        cam_target = np.array([char_pos[0], char_pos[1], 1.0])

        cam_mode = camera.CameraMode[env_config["camera_mode"]]
        self._camera = camera.Camera(mode=cam_mode,
                                     engine=self._engine,
                                     pos=cam_pos,
                                     target=cam_target,
                                     track_env_id=env_id,
                                     track_obj_id=char_id)
        return
    
    def _get_char_color(self):
        engine_name = self._engine.get_name()
        if (engine_name == "isaac_lab"):
            col = np.array([0.2, 0.25, 0.7])
        elif (engine_name == "newton"):
            col = np.array([0.35, 0.45, 0.7])
        else:
            col = np.array([0.5, 0.65, 0.95])
        return col


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def convert_to_local_body_pos(root_rot, body_pos):
    # type: (Tensor, Tensor) -> Tensor
    
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
    heading_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_expand = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                            heading_rot_expand.shape[2])
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_local_body_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])

    return local_body_pos

@torch.jit.script
def convert_to_local_root_body_pos(root_rot, body_pos):
    # type: (Tensor, Tensor) -> Tensor
    
    root_inv_rot = torch_util.quat_conjugate(root_rot)
    root_rot_expand = root_inv_rot.unsqueeze(-2)
    root_rot_expand = root_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_root_rot_expand = root_rot_expand.reshape(root_rot_expand.shape[0] * root_rot_expand.shape[1], 
                                                   root_rot_expand.shape[2])
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_local_body_pos = torch_util.quat_rotate(flat_root_rot_expand, flat_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])

    return local_body_pos

# @torch.jit.script
# def compute_char_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot,
#                      dof_vel, key_pos, global_obs, root_height_obs, cmds,
#                      asymmetric_obs):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, bool) -> Tuple[Tensor, Tensor]
#     heading_rot = torch_util.calc_heading_quat_inv(root_rot)
#
#     if (global_obs):                            # <=
#         root_rot_obs = torch_util.quat_to_tan_norm(root_rot)
#         root_vel_obs = root_vel
#         root_ang_vel_obs = root_ang_vel
#     else:
#         local_root_rot = torch_util.quat_mul(heading_rot, root_rot)
#         root_rot_obs = torch_util.quat_to_tan_norm(local_root_rot)
#         root_vel_obs = torch_util.quat_rotate(heading_rot, root_vel)
#         root_ang_vel_obs = torch_util.quat_rotate(heading_rot, root_ang_vel)
#
#     joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1], joint_rot.shape[2]])
#     joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
#     joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1] * joint_rot_obs_flat.shape[-1]])
#
#     # obs = [root_rot_obs, root_vel_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]
#     # 1. 组装特权/全量观测 (Critic用)，保持原有的组合顺序
#     full_obs_list = [root_rot_obs, root_vel_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]
#
#     if (len(key_pos) > 0):
#         root_pos_expand = root_pos.unsqueeze(-2)
#         key_pos = key_pos - root_pos_expand
#         if (not global_obs):    # ×
#             key_pos = convert_to_local_body_pos(root_rot, key_pos)
#
#         key_pos_flat = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1] * key_pos.shape[2]])
#         # obs = obs + [key_pos_flat]
#         full_obs_list.append(key_pos_flat)
#
#     if (root_height_obs):   # <=
#         root_h = root_pos[:, 2:3]
#         # obs = [root_h] + obs
#         full_obs_list = [root_h] + full_obs_list
#     if len(cmds)>0:
#         full_obs_list.append(cmds)
#     # obs = torch.cat(obs, dim=-1)
#     full_obs_tensor = torch.cat(full_obs_list, dim=-1)
#
#     # 2. 根据 asymmetric_obs 开关决定 Actor 观测
#     if asymmetric_obs:
#         # 无特权观测 (用于部署): 仅保留 root_rot, root_ang_vel, joint_rot, dof_vel
#         actor_obs_list = [root_rot_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]
#         if len(cmds)>0:
#             actor_obs_list.append(cmds)
#         actor_obs_tensor = torch.cat(actor_obs_list, dim=-1)
#     else:
#         # 对称模式：Actor 也能看到全部特权信息
#         actor_obs_tensor = full_obs_tensor
#
#     # return obs
#     return actor_obs_tensor, full_obs_tensor


@torch.jit.script
def compute_char_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot,
                     dof_vel, key_pos, global_obs, root_height_obs, cmds, push,
                     asymmetric_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor, bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    heading_rot = torch_util.calc_heading_quat_inv(root_rot)

    if (global_obs):  # <=
        root_rot_obs = torch_util.quat_to_tan_norm(root_rot)
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel
    else:
        local_root_rot = torch_util.quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_util.quat_to_tan_norm(local_root_rot)
        root_vel_obs = torch_util.quat_rotate(heading_rot, root_vel)
        root_ang_vel_obs = torch_util.quat_rotate(heading_rot, root_ang_vel)

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1], joint_rot.shape[2]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat,
                                  [joint_rot.shape[0], joint_rot.shape[1] * joint_rot_obs_flat.shape[-1]])

    # obs = [root_rot_obs, root_vel_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]
    # 1. 组装特权/全量观测 (Critic用)，保持原有的组合顺序
    full_obs_list = [root_rot_obs, root_vel_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]

    if (len(key_pos) > 0):
        root_pos_expand = root_pos.unsqueeze(-2)
        key_pos = key_pos - root_pos_expand
        if (not global_obs):  # ×
            key_pos = convert_to_local_body_pos(root_rot, key_pos)

        key_pos_flat = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1] * key_pos.shape[2]])
        # obs = obs + [key_pos_flat]
        full_obs_list.append(key_pos_flat)
    else:
        key_pos_flat = torch.zeros((root_pos.shape[0], 0), device=root_pos.device)

    if (root_height_obs):  # <=
        root_h = root_pos[:, 2:3]
        # obs = [root_h] + obs
        full_obs_list = [root_h] + full_obs_list
    if len(cmds) > 0:
        full_obs_list.append(cmds)
    if len(push) > 0:
        full_obs_list.append(push)
    full_obs_tensor = torch.cat(full_obs_list, dim=-1)

    # 2. 单独抽取 prop (本体感知)
    prop_obs_list = [root_rot_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]
    if len(cmds) > 0:
        prop_obs_list.append(cmds)
    prop_obs_tensor = torch.cat(prop_obs_list, dim=-1)
    # 3. 单独抽取 priv (特权信息)
    priv_obs_list = []
    if (root_height_obs):
        priv_obs_list.append(root_pos[:, 2:3])
    priv_obs_list.append(root_vel_obs)
    if (len(key_pos) > 0):
        priv_obs_list.append(key_pos_flat)
    if len(push) > 0:
        priv_obs_list.append(push)
    priv_obs_tensor = torch.cat(priv_obs_list, dim=-1) if len(priv_obs_list) > 0 else torch.zeros(
        (root_pos.shape[0], 0), device=root_pos.device)

    # 2. 根据 asymmetric_obs 开关决定 Actor 观测
    if asymmetric_obs:
        # 无特权观测 (用于部署): 仅保留 root_rot, root_ang_vel, joint_rot, dof_vel
        actor_obs_tensor = prop_obs_tensor
    else:
        # 对称模式：Actor 也能看到全部特权信息
        actor_obs_tensor = full_obs_tensor

    # return obs
    return actor_obs_tensor, full_obs_tensor, prop_obs_tensor, priv_obs_tensor


@torch.jit.script
def compute_reward(root_pos):
    # type: (Tensor) -> Tensor
    r = torch.ones_like(root_pos[..., 0])
    return r

@torch.jit.script
def compute_done(done_buf, time, ep_len):
    # type: (Tensor, Tensor, float) -> Tensor
    timeout = time >= ep_len
    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)
    done[timeout] = base_env.DoneFlags.TIME.value
    return done