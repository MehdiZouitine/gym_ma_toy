import gym
import gym_ma_toy

if __name__ == "__main__":
    env = gym.make("team_catcher-v0")

    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()
