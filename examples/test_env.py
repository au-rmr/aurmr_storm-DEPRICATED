from storm_kit.gym.core import Gym
import argparse
import arm_training_env
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path


def run_env(env):
    while True:
        try:
            env.step(1)
        except KeyboardInterrupt:
            print('Closing')
            done = True
            break
    
    env.close()

        # state = env.resetnvidia()
        # if render:
        #     env.render()

        # for step in range(2000):
        #     action = policy(state)

        #     state, reward, done, _ = env.step(action)

        #     if render:
        #         env.render()




if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='ur16e', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    parser.add_argument('--camera_pose', nargs='+', type=float, default=[2.0, 0.0, 0.0, 0.707,0.0,0.0,-0.707], help='Where to spawn camera')
    parser.add_argument('--num_env', type=int, default='1', help='Number of environments')
    args = parser.parse_args()
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless
    #sim_params['up_axis'] = gymapi.UP_AXIS_Z
    gym_instance = Gym(**sim_params)
    
    env = arm_training_env.TahomaEnv(args, gym_instance)

    run_env(env)