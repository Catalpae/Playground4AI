import gym
import torch
import argparse

parser = argparse.ArgumentParser(description='env settings')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=True, help='render the environment')
args = parser.parse_args()

env = gym.make('Pendulum-v1', render_mode="human").unwrapped
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
torch.manual_seed(args.seed)


def main():
    for i_epoch in range(1000):
        state, info = env.reset()
        if args.render:
            env.render()

        for t in range(200):
            next_state, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if args.render:
                env.render()

            if terminated or truncated:
                break
    env.close()


if __name__ == '__main__':
    main()

