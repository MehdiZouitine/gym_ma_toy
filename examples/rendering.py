import gym
import gym_ma_toy
import time

VERSION = "v2"

if __name__ == "__main__":
    env = gym.make("team_catcher-" + VERSION)

    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        time.sleep(0.1)
    env.close()
