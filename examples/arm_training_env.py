import copy
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymapi

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import time
import yaml
import argparse
import numpy as np
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt

from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask
np.set_printoptions(precision=2)

from typing import Union

class IsaacGymEnv():
    def __init__(self, args, gym_instance):

        self.robot_file = args.robot + '.yml'
        self.task_file = args.robot + '_reacher_collision.yml'
        self.world_file = 'collision_primitives_3d_collision.yml'

        self.gym_instance = gym_instance
        self.gym = gym_instance.gym
        self.sim = gym_instance.sim
        
        self.world_yml = join_path(get_gym_configs_path(), self.world_file)
        with open(self.world_yml) as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)

        self.robot_yml = join_path(get_gym_configs_path(),args.robot + '.yml')
        with open(self.robot_yml) as file:
            self.robot_params = yaml.load(file, Loader=yaml.FullLoader)
        self.sim_params = self.robot_params['sim_params']
        self.sim_params['asset_root'] = get_assets_path()
        if(args.cuda):
            device = 'cuda'
        else:
            device = 'cpu'
        self.sim_params['collision_model'] = None
        
        # create robot simulation:
        self.robot_sim = RobotSim(gym_instance=self.gym, sim_instance=self.sim, **self.sim_params, device=device)

        # create gym environment:
        self.robot_pose = self.sim_params['robot_pose']
        self.env_ptr = gym_instance.env_list[0]
        self.robot_ptr = self.robot_sim.spawn_robot(self.env_ptr, self.robot_pose, coll_id=2)
        
        self.tensor_args = {'device':torch.device('cuda', 0) , 'dtype':torch.float32}

        # spawn camera:
        self.robot_sim.spawn_camera(self.env_ptr, 60, 640, 480, np.array(args.camera_pose)) 
    
        # get pose
        self.w_T_r = copy.deepcopy(self.robot_sim.spawn_robot_pose)
        
        self.w_T_robot = torch.eye(4)
        quat = torch.tensor([self.w_T_r.r.w,self.w_T_r.r.x,self.w_T_r.r.y,self.w_T_r.r.z]).unsqueeze(0)
        rot = quaternion_to_matrix(quat)
        self.w_T_robot[0,3] = self.w_T_r.p.x
        self.w_T_robot[1,3] = self.w_T_r.p.y
        self.w_T_robot[2,3] = self.w_T_r.p.z
        self.w_T_robot[:3,:3] = rot[0]

        # World Instance
        self.world_instance = World(self.gym, self.sim, self.env_ptr, self.world_params, w_T_r=self.w_T_r)   

        # Control Parameters
        self.mpc_control = ReacherTask(self.task_file, self.robot_file, self.world_file, self.tensor_args)
        self.sim_dt = self.mpc_control.exp_params['control_dt']

        self.mpc_control.update_params(goal_state=np.array([-0.85, 0.6, 0.2, -1.8, 0.0, 2.4, 0.0,
                                    10.0, 10.0, 10.0,  0.0, 0.0, 0.0, 0.0]))

        # Simulation time step
        self.t_step = 0

class TahomaEnv(IsaacGymEnv):
    def __init__(self, args, gym_instance):
        super().__init__(args, gym_instance)
        self.ee_pose = gymapi.Transform()



    def step(self, action):
        self.gym_instance.step()
        self.t_step += self.sim_dt
        pose_reached = self.pose_reached()
        if (pose_reached): print('######################REACHED#####################')
        self.set_goal(np.array([0.5, 1.2, 0.0, 0,0.707,0, 0.707]))
        q_des, qd_des, qdd_des = self.move_robot()
        self.draw_lines()
        # done = np.array([False, False])
        # reward = self.get_reward(pose_reached, action)
        # ob = self.get_obs()
        # return ob, reward, done, None
    
    def close(self):
        self.mpc_control.close()

    def set_goal(self, pose:np.ndarray):
        goal_pose = gymapi.Transform()
        goal_pose.p = gymapi.Vec3(pose[0], pose[1], pose[2]) 
        goal_pose.r = gymapi.Quat(pose[3], pose[4], pose[5], pose[6])
        pose = copy.deepcopy(self.w_T_r.inverse() * goal_pose)

        g_pos = np.zeros(3)
        g_q = np.zeros(4)
        g_pos[0] = pose.p.x
        g_pos[1] = pose.p.y
        g_pos[2] = pose.p.z
        g_q[0] = pose.r.w
        g_q[1] = pose.r.x
        g_q[2] = pose.r.y
        g_q[3] = pose.r.z

        self.mpc_control.update_params(goal_ee_pos=g_pos, goal_ee_quat=g_q)


    # Output: Returns if goal pose was reached
    # TODO Include Quaternion 
    def pose_reached(self)->bool:
        g_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        g_q = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())

        current_robot_state = copy.deepcopy(self.robot_sim.get_state(self.env_ptr, self.robot_ptr)) 
        filtered_state_mpc = current_robot_state
        curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
        curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0)
        pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
        e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
        e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
        self.ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
        self.ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])

        test = np.linalg.norm(g_pos - np.ravel([self.ee_pose.p.x, self.ee_pose.p.y, self.ee_pose.p.z]))
        print(test)
        return test < 0.002
        #return np.linalg.norm(g_pos - np.ravel([self.ee_pose.p.x, self.ee_pose.p.y, self.ee_pose.p.z])) < 0.002# or (np.linalg.norm(g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z]))<0.1)

    def move_robot(self) -> Union[np.array, np.array, np.array]:
        current_robot_state = copy.deepcopy(self.robot_sim.get_state(self.env_ptr, self.robot_ptr))
        command = self.mpc_control.get_command(self.t_step, current_robot_state, control_dt=self.sim_dt, WAIT=False)
        q_des = copy.deepcopy(command['position'])
        qd_des = copy.deepcopy(command['velocity'])
        qdd_des = copy.deepcopy(command['acceleration'])
        self.robot_sim.command_robot_position(q_des, self.env_ptr, self.robot_ptr)
        return q_des, qd_des, qdd_des

    def draw_lines(self):
        self.gym_instance.clear_lines()
        w_robot_coord = CoordinateTransform(trans=self.w_T_robot[0:3,3].unsqueeze(0),
                                        rot=self.w_T_robot[0:3,0:3].unsqueeze(0))
        top_trajs = self.mpc_control.top_trajs.cpu().float()
        n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
        w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)

        top_trajs = w_pts.cpu().numpy()
        color = np.array([0.0, 1.0, 0.0])
        for k in range(top_trajs.shape[0]):
            pts = top_trajs[k,:,:]
            color[0] = float(k) / float(top_trajs.shape[0])
            color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
            self.gym_instance.draw_lines(pts, color=color)
