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



class IsaacGymEnv():
    def __init__(self, args, gym_instance):
        self.vis_ee_target = True
        self.robot_file = args.robot + '.yml'
        self.task_file = args.robot + '_reacher_collision.yml'
        self.world_file = 'collision_primitives_3d_collision.yml'

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
        
        tensor_args = {'device':torch.device('cuda', 0) , 'dtype':torch.float32}

        # spawn camera:
        self.robot_sim.spawn_camera(self.env_ptr, 60, 640, 480, np.array(args.camera_pose)) 
    
        # get pose
        w_T_r = copy.deepcopy(self.robot_sim.spawn_robot_pose)
        
        w_T_robot = torch.eye(4)
        quat = torch.tensor([w_T_r.r.w,w_T_r.r.x,w_T_r.r.y,w_T_r.r.z]).unsqueeze(0)
        rot = quaternion_to_matrix(quat)
        w_T_robot[0,3] = w_T_r.p.x
        w_T_robot[1,3] = w_T_r.p.y
        w_T_robot[2,3] = w_T_r.p.z
        w_T_robot[:3,:3] = rot[0]

        # World Instance
        self.world_instance = World(self.gym, self.sim, self.env_ptr, self.world_params, w_T_r=w_T_r)   

        # Control Parameters
        mpc_control = ReacherTask(self.task_file, self.robot_file, self.world_file, tensor_args)

        self.n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
        

        mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}

        mpc_control.update_params(goal_state=np.array([-0.85, 0.6, 0.2, -1.8, 0.0, 2.4, 0.0,
                                    10.0, 10.0, 10.0,  0.0, 0.0, 0.0, 0.0]))
                                    
        
        ee_error = 10.0
        j = 0
        t_step = 0
        i = 0

