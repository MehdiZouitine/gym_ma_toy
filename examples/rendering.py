import gymnasium as gym
import gym_ma_toy
import time

VERSION = "v2"

if __name__ == "__main__":
    env = gym.make("team_catcher-" + VERSION)

    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        time.sleep(0.1)
    env.close()
