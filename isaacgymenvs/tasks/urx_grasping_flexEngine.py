import numpy as np
import os
import torch
import random
import math

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

# @torch.jit.script
# def quat_axis(q, axis=0):
#     basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
#     basis_vec[:, axis] = 1
#     return quat_rotate(q, basis_vec)
#
@torch.jit.script
def orientation_error(desired, current):
    # type: (Tensor, Tensor) -> Tensor

    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def control_ik(dpose, j_eef, num_envs):
    # type: (Tensor, Tensor, int) -> Tensor
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=dpose.device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
    return u

class URxGraspingFlex(VecTask):
    # TODO:it is not useful, and building a simple reachable task now.
    def __init__(self, cfg, rl_device, sim_device,
                 graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.urx_position_noise = self.cfg["env"]["urxPositionNoise"]
        self.urx_rotation_noise = self.cfg["env"]["urxRotationNoise"]
        self.urx_dof_noise = self.cfg["env"]["urxDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"] # a means for the physics engine to optimize things better for performance

        # self.reset_dist = self.cfg["env"]["resetDist"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"}, \
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cube_pose (7) + eef_pose (7) + q_gripper (1)
        self.cfg["env"]["numObservations"] = 7 if self.control_type == "osc" else 14
        # actions include: delta EEF if OSC (6) or joint torques (6) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}  # will be dict filled with relevant states to use for reward calculation
        self.handles = {}  # will be dict mapping names to relevant sim handles
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed
        self._init_box_state = None  # Initial state of cube for the current env
        self._box_state = None  # Current state of cube for the current env
        self._box_id = None  # Actor ID corresponding to cube for a given env

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None  # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._urx_effort_limits = None  # Actuator effort limits for franka
        self._global_indices = None  # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # URx defaults
        self.urx_default_dof_pos = to_torch(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=self.device
        )

        # Gripper defaults
        self.gripper_default_dof_pos = to_torch(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Set control limits, xyz+rpy
        self.cmd_limit = to_torch([0.05, 0.05, 0.05, 0.1, 0.1, 0.1], device=self.device).unsqueeze(0) if \
            self.control_type == "osc" else self._urx_effort_limits[:6].unsqueeze(0)

        self.device = "cpu"
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.physics_engine = gymapi.SIM_FLEX
        if self.physics_engine != gymapi.SIM_FLEX:
            print("*** This project only supports FleX")
            quit()
        self.sim_params.flex.solver_type = 5
        self.sim_params.flex.num_outer_iterations = 4
        self.sim_params.flex.num_inner_iterations = 20
        self.sim_params.flex.relaxation = 0.8
        self.sim_params.flex.warm_start = 0.75
        self.sim_params.flex.shape_collision_margin = 5e-4
        self.sim_params.flex.contact_regularization = 1.0e-6
        self.sim_params.flex.shape_collision_margin = 0.1
        self.sim_params.flex.deterministic_mode = True

        # enable Von-Mises stress visualization
        self.sim_params.stress_visualization = True
        self.sim_params.stress_visualization_min = 0.0
        self.sim_params.stress_visualization_max = 1.e+5

        self.sim_params.use_gpu_pipeline = False
        self.device = "cpu"
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        # urx_asset_file = "urdf/ur5e/ur5e_gripper.urdf"
        gripper_asset_file = "urdf/robotiq_arg85_description/robots/robotiq_arg85_description.urdf"
        soft_asset_file = "urdf/softbody.urdf"

        soft_thickness = 0.1  # important to add some thickness to the soft body to avoid interpenetrations

        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #     # urx_asset_file = self.cfg["env"]["asset"].get("assetFileNameURx", urx_asset_file)
        #     gripper_asset_file = self.cfg["env"]["asset"].get("assetFileNameURx", gripper_asset_file) # TODO:this need to be modified

        #load gripper asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        # asset_options.use_mesh_materials = True
        # asset_options.override_inertia = True
        # asset_options.override_com = True
        # asset_options.vhacd_enabled = True
        asset_options.thickness = soft_thickness
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.flip_visual_attachments = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # asset_options.use_mesh_materials = True
        gripper_asset = self.gym.load_asset(self.sim, asset_root, gripper_asset_file, asset_options)
        # num_dof = self.gym.get_asset_dof_count(gripper_asset)
        # self.hand_index = self.gym.get_asset_rigid_body_dict(gripper_asset)["dummy_center_indicator_link"]
        # gripper_dofs = 6
        # self.urx_arm_dofs = num_dof - gripper_dofs

        # urx_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 200, 200., 200., 200, 200., 200.], dtype=torch.float, device=self.device)
        # urx_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 200, 200., 200., 200, 200., 200.], dtype=torch.float, device=self.device)

        self.num_gripper_bodies = self.gym.get_asset_rigid_body_count(gripper_asset)
        self.num_gripper_dofs = self.gym.get_asset_dof_count(gripper_asset)

        print("num gripper bodies: ", self.num_gripper_bodies)
        print("num gripper dofs: ", self.num_gripper_dofs)

        gripper_dof_props = self.gym.get_asset_dof_properties(gripper_asset)

        # building soft body
        print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
        soft_asset = self.gym.load_asset(self.sim, asset_root, soft_asset_file, asset_options)
        asset_soft_body_count = self.gym.get_asset_soft_body_count(soft_asset)
        asset_soft_materials = self.gym.get_asset_soft_materials(soft_asset)
        # Print asset soft material properties
        print('Soft Material Properties:')
        for i in range(asset_soft_body_count):
            mat = asset_soft_materials[i]
            print(f'(Body {i}) youngs: {mat.youngs} poissons: {mat.poissons} damping: {mat.damping}')

        self.gripper_dof_lower_limits = []
        self.gripper_dof_upper_limits = []
        gripper_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        if self.physics_engine == gymapi.SIM_FLEX:
            gripper_dof_props["stiffness"][:].fill(500.0)
            gripper_dof_props["damping"][:].fill(40.0)
        else:
            print("*** This project only supports FleX")
            quit()

        self.gripper_dof_lower_limits = gripper_dof_props["lower"]
        self.gripper_dof_upper_limits = gripper_dof_props["upper"]
        self.gripper_dof_lower_limits = to_torch(self.gripper_dof_lower_limits, device=self.device)
        self.gripper_dof_upper_limits = to_torch(self.gripper_dof_upper_limits, device=self.device)
        gripper_dof_props['effort'][:].fill(200)

        # # disable collision
        # urx_shape_props = self.gym.get_actor_rigid_shape_properties(urx_asset)
        # for i in range(len(urx_shape_props)):
        #     urx_shape_props[i].friction = 0

        # Define start pose for gripper
        gripper_start_pose = gymapi.Transform()
        gripper_start_pose.p = gymapi.Vec3(0, 0.0, 1)
        gripper_start_pose.r = gymapi.Quat(0.0, 1.0, 0.0, 0.0) # xyzw

        # compute aggregate size
        num_gripper_bodies = self.gym.get_asset_rigid_body_count(gripper_asset)
        num_gripper_shapes = self.gym.get_asset_rigid_shape_count(gripper_asset)
        max_agg_bodies = num_gripper_bodies + 1 # 1 for box
        max_agg_shapes = num_gripper_shapes + 1 # 1 for box

        # create box asset
        self.box_size = 0.025
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        box_asset = self.gym.create_box(self.sim, *([self.box_size] * 3), asset_options)
        box_color = gymapi.Vec3(0.0, 0.4, 0.1)
        box_pose = gymapi.Transform()

        # create table asset
        table_dims = gymapi.Vec3(0.6, 1.0, 0.1)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

        # soft body pose
        softbody_pose = gymapi.Transform()
        # softbody_pose.p.z = 0.5
        softbody_pose.r = gymapi.Quat(0, 0, 0, 1)

        self.box_idxs = []
        self.urxs = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []
        self.init_pos_list = []
        self.init_rot_list = []
        self.hand_idxs = []
        self.soft_actors = []

        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: URx should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create gripper
            if self.urx_position_noise > 0:
                rand_xy = self.urx_position_noise * (-1. + np.random.rand(2) * 2.0)
                gripper_start_pose.p = gymapi.Vec3(-0.0 + rand_xy[0], 0.0 + rand_xy[1], 1.0)
            if self.urx_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.urx_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                gripper_start_pose.r = gymapi.Quat(*new_quat)
            gripper_actor = self.gym.create_actor(env_ptr, gripper_asset, gripper_start_pose, "gripper", i, 1)
            self.gym.set_actor_dof_properties(env_ptr, gripper_actor, gripper_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor)
            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            # add soft body + rail actor
            softbody_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
            softbody_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
            softbody_pose.p.z = table_dims.z + 0.01
            soft_actor = self.gym.create_actor(env_ptr, soft_asset, softbody_pose, "soft", i, 0)
            # set soft material within a range of default
            actor_default_soft_materials = self.gym.get_actor_soft_materials(env_ptr, soft_actor)
            actor_soft_materials = self.gym.get_actor_soft_materials(env_ptr, soft_actor)
            for j in range(asset_soft_body_count):
                youngs = actor_soft_materials[j].youngs
                actor_soft_materials[j].youngs = random.uniform(youngs * 0.2, youngs * 2.4)

                poissons = actor_soft_materials[j].poissons
                actor_soft_materials[j].poissons = random.uniform(poissons * 0.8, poissons * 1.2)

                damping = actor_soft_materials[j].damping
                # damping is 0, instead we just randomize from scratch
                actor_soft_materials[j].damping = random.uniform(0.0, 0.00008) ** 2

                self.gym.set_actor_soft_materials(env_ptr, soft_actor, actor_soft_materials)



            # # Create URx
            # # Potentially randomize start pose, donot randmize the rotation
            # if self.urx_position_noise > 0:
            #     rand_xy = self.urx_position_noise * (-1. + np.random.rand(2) * 2.0)
            #     urx_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1], 1.0)
            # if self.urx_rotation_noise > 0:
            #     rand_rot = torch.zeros(1, 3)
            #     rand_rot[:, -1] = self.urx_rotation_noise * (-1. + np.random.rand() * 2.0)
            #     new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
            #     urx_start_pose.r = gymapi.Quat(*new_quat)
            # urx_actor = self.gym.create_actor(env_ptr, urx_asset, urx_start_pose, "urx", i, 3)
            # self.gym.set_actor_dof_properties(env_ptr, urx_actor, urx_dof_props)
            # # self.gym.set_actor_rigid_shape_properties(env_ptr, urx_actor, urx_shape_props)
            #
            # if self.aggregate_mode == 2:
            #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            #
            # if self.aggregate_mode == 1:
            #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            #
            # self._box_id = self.gym.create_actor(env_ptr, box_asset, box_pose, "box", i, 3)
            # self.gym.set_rigid_body_color(env_ptr, self._box_id, 0, gymapi.MESH_VISUAL, box_color)
            # # box_idx = self.gym.get_actor_rigid_body_index(env_ptr, self._box_id, 0, gymapi.DOMAIN_SIM)
            #
            # if self.aggregate_mode > 0:
            #     self.gym.end_aggregate(env_ptr)
            #
            # hand_idx = self.gym.find_actor_rigid_body_index(env_ptr, urx_actor, "tool0", gymapi.DOMAIN_SIM)
            # self.hand_idxs.append(hand_idx)
            self.envs.append(env_ptr)
            self.soft_actors.append(soft_actor)
            # self.urxs.append(urx_actor)
            # # self.box_idxs.append(box_idx)

        self._init_box_state = torch.zeros(self.num_envs, 13, device=self.device)
        # x y z + rpy + vxyz + rvxyz
        self.init_data()

    def init_data(self):
        # Setup sim handles
        # print(self.envs, "-----------------")
        env_ptr = self.envs[0]

        # urx_handle = 0
        # self.handles = {
        #     # URx
        #     "hand": self.gym.find_actor_rigid_body_handle(env_ptr, urx_handle, "dummy_center_indicator_link"),
        #     # Cubes
        #     "box_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._box_id, "box"),
        # }

        gripper_handle = 0
        self.handles = {
            # URx
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, gripper_handle, "robotiq_85_base_link"),
            # Cubes
            # "box_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._box_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.rb_states = gymtorch.wrap_tensor(_rigid_body_state_tensor)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["hand"], :]
        # self._meef_state = self.rb_states[self.hand_idxs, :]

        # _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "urx")
        # jacobian = gymtorch.wrap_tensor(_jacobian)
        # # hand_index = self.gym.get_asset_rigid_body_dict(env_ptr, urx_handle)['dummy_center_indicator_link']
        # self._j_eef = jacobian[:, self.hand_index - 1, :, :6]
        # # _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "urx")
        # # mm = gymtorch.wrap_tensor(_massmatrix)
        # # self._mm = mm[:, :6, :6]
        # self._box_state = self._root_state[:, self._box_id, :]
        # #
        # # # Initialize states
        # # self.states.update({
        # #     "box_size": torch.ones_like(self._eef_state[:, 0]) * self.box_size,
        # # })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :6]
        self._gripper_control = self._pos_control[:, 6:12]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        # self.states.update({
        #     "q": self._q[:, :],
        #     "eef_pos": self._eef_state[:, :3],
        #     "eef_quat": self._eef_state[:, 3:7],
        #     "box_quat": self._box_state[:, 3:7],
        #     "box_pos": self._box_state[:, :3],
        #     "box_pos_relative": self._box_state[:, :3] - self._eef_state[:, :3],
        # })

        self.states.update({
            "q": self._q[:, :],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
        })
        # print("hand index:", self.hand_idxs)
        # print("rb_states shape:", self.rb_states.shape)
        # print("eef pos:", self.states['eef_pos'][0, :])
        # print("box pos:", self.states['box_pos'][0, :])

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_urx_reward(
            self.reset_buf, self.progress_buf,
            self.states,
            self.max_episode_length
           )

    def compute_observations(self, env_ids=None):
        self._refresh()
        obs = ["eef_pos", "eef_quat"] # 4 + 3 + 3 + 4 = 14
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset box
        # self._reset_init_cube_state(env_ids=env_ids, check_valid=True)
        # self._box_state[env_ids] = self._init_box_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 12), device=self.device)
        # print("------------", len(env_ids))
        # print(self.gripper_default_dof_pos, "-----------")
        # print(self.gripper_dof_lower_limits, "-----------")
        # print(self.gripper_dof_upper_limits, "-----------")
        pos = tensor_clamp(
            self.gripper_default_dof_pos.unsqueeze(0),
            self.gripper_dof_lower_limits.unsqueeze(0),
            self.gripper_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -6:] = self.urx_default_dof_pos[-6:]

        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, 1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _reset_init_cube_state(self, env_ids, check_valid=True):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        this_box_state_all = self._init_box_state
        sampled_cube_state[:, 2] = 1.5 # Set z value, which is fixed height
        sampled_cube_state[:, 6] = 1.0 # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, :2] = 1.5 * self.start_position_noise * \
                                    (torch.rand(num_resets, 2, device=self.device) - 0.5)
        this_box_state_all[env_ids, :] = sampled_cube_state

    def pre_physics_step(self, actions):
        # print("actions:", actions)
        self.actions = actions.clone().to(self.device)

        # # Split arm and gripper command
        # u_arm, u_gripper = self.actions[:, :6], self.actions[:, -1]
        #
        # u_arm = u_arm * self.cmd_limit / self.action_scale
        # # self._arm_control[:, :] = self.states['eef_pos'] + u_arm
        # ee_orn_err = u_arm[:, 3:]
        # ee_pos_err = u_arm[:, :3]
        # dpose = torch.cat([ee_pos_err, ee_orn_err], -1).unsqueeze(-1)
        # self._pos_control[:, :self.urx_arm_dofs] = self.states['q'][:, :6] + \
        #                                            control_ik(dpose, self._j_eef, self.num_envs)
        # # print(control_ik(dpose, self._j_eef, self.num_envs))
        # u_fingers = torch.zeros_like(self._gripper_control)
        # self._gripper_control[:, :] = u_fingers
        # # Deploy actions
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

        # # Control arm (scale value first)
        # u_arm = u_arm * self.cmd_limit / self.action_scale
        # last_q = self.states['q'][:, :6]
        # # print(self._pos_control[0, :6])
        # self._arm_control[:, :] = u_arm + last_q
        # # print("arm control:", self._arm_control[0, :])
        # # Control gripper
        # u_fingers = torch.zeros_like(self._gripper_control)
        # self._gripper_control[:, :] = u_fingers
        # self._pos_control[:, :6] = torch.clone(self._arm_control[:, :6])
        # # Deploy actions
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_urx_reward(
        reset_buf, progress_buf,
        states,
        max_episode_length
):
    # type: (Tensor, Tensor, Dict[str, Tensor], float) -> Tuple[Tensor, Tensor]

    # self.states.update({
    #     # Franka
    #     "q": self._q[:, :],
    #     "eef_pos": self._eef_state[:, :3],
    #     "eef_quat": self._eef_state[:, 3:7],
    #     # Cubes
    #     "box_quat": self._box_state[:, 3:7],
    #     "box_pos": self._box_state[:, :3],
    #     "box_pos_relative": self._box_state[:, :3] - self._eef_state[:, :3],
    # })

    # d = torch.norm(states["cubeA_pos_relative"], dim=-1)
    # reward = 1 - torch.tanh(10.0 * d)
    d = torch.norm(states["eef_pos"], dim=-1)
    rewards = 1 - torch.tanh(10.0 * d)
    # print(states["box_pos"])
    # print("dis rewrad:", -d)
    # reward = -torch.norm(grasp_pos - target_pos, p=2, dim=-1)
    # reset_buf = torch.where(torch.abs(d) < 0.01, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (torch.abs(d) < 0.01), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf
