from gym_ma_toy.envs.coop.game_base import WorldBase
from gym_ma_toy.envs.coop.team_catcher_base import TeamCatcherBase
import numpy as np

class TestGame:
    def test_attribute(self):
        size_list = [10, 20, 100, 200, 1000, 2000, 3000]
        nb_agents_hv_list = [2, 4, 10, 20, 100, 200, 1000]
        nb_agents_diag_list = [2, 4, 10, 20, 100, 200, 1000]
        nb_targets_list = [2, 4, 10, 20, 100, 200, 1000]
        nb_mobiles_list = [2, 4, 10, 20, 100, 200, 1000]
        fow_agents_hv_list = [0,1,2,3,4,10,15,20,25]
        fow_agents_diag_list = [0,1,2,3,4,10,15,20,25]
        worlds = [
            WorldBase(
                size=size_list[i],
                nb_agents_hv=nb_agents_hv_list[i],
                nb_agents_diag=nb_agents_diag_list[i],
                nb_targets=nb_targets_list[i],
                nb_mobiles=nb_mobiles_list[i],
                fow_agents_hv=fow_agents_hv_list[i],
                fow_agents_diag=fow_agents_diag_list[i],
                seed=7,
            )
            for i in range(len(size_list))
        ]
        for i in range(len(size_list)):
            assert (
                (worlds[i].size == size_list[i])
                and (worlds[i].nb_agents_hv == nb_agents_hv_list[i])
                and (worlds[i].nb_agents_diag == nb_agents_diag_list[i])
                and (worlds[i].nb_targets == nb_targets_list[i])
                and (worlds[i].nb_mobiles == nb_mobiles_list[i])
                and (worlds[i].fow_agents_hv == fow_agents_hv_list[i])
                and (worlds[i].fow_agents_diag == fow_agents_diag_list[i])
                and (worlds[i].seed == 7)
            )

    def test_reset(self):
        size_list = [10, 20, 100, 200, 1000, 2000, 3000]
        nb_agents_hv_list = [2, 4, 10, 20, 100, 200, 1000]
        nb_agents_diag_list = [2, 4, 10, 20, 100, 200, 1000]
        nb_targets_list = [2, 4, 10, 20, 100, 200, 1000]
        nb_mobiles_list = [2, 4, 10, 20, 100, 200, 1000]
        fow_agents_hv_list = [0,1,2,3,4,10,15,20,25]
        fow_agents_diag_list = [0,1,2,3,4,10,15,20,25]
        worlds = [
            WorldBase(
                size=size_list[i],
                nb_agents_hv=nb_agents_hv_list[i],
                nb_agents_diag=nb_agents_diag_list[i],
                nb_targets=nb_targets_list[i],
                nb_mobiles=nb_mobiles_list[i],
                fow_agents_hv=fow_agents_hv_list[i],
                fow_agents_diag=fow_agents_diag_list[i],
                seed=7,
            )
            for i in range(len(size_list))
        ]
        for i in range(len(worlds)):
            worlds[i].reset()
            assert (
                (worlds[i].size == size_list[i])
                and (worlds[i].nb_agents_hv == nb_agents_hv_list[i])
                and (worlds[i].nb_agents_diag == nb_agents_diag_list[i])
                and (worlds[i].nb_targets == nb_targets_list[i])
                and (worlds[i].nb_mobiles == nb_mobiles_list[i])
                and (worlds[i].fow_agents_hv == fow_agents_hv_list[i])
                and (worlds[i].fow_agents_diag == fow_agents_diag_list[i])
                and (worlds[i].seed == 7)
            )

    def check_position_unicity(world):
        agents = world.agents #dict
        targets = world.targets #deque
        mobiles = world.mobiles #deque
        all_position = []
        for _,agent in agents.items():
            all_position.append(agent.position)
        for target in targets:
            all_position.append(target.position)
        for mobile in mobiles:
            all_position.append(mobile.position)
        
        assert len(list(set(all_position))) == len(all_position)

        

    def test_position(self):
        size_list = [10, 20, 100, 200, 1000, 2000, 3000]
        nb_agents_hv_list = [2, 4, 10, 20, 100, 200, 1000]
        nb_agents_diag_list = [2, 4, 10, 20, 100, 200, 1000]
        nb_targets_list = [2, 4, 10, 20, 100, 200, 1000]
        nb_mobiles_list = [2, 4, 10, 20, 100, 200, 1000]
        fow_agents_hv_list = [0,1,2,3,4,10,15,20,25]
        fow_agents_diag_list = [0,1,2,3,4,10,15,20,25]
        envs = [
            TeamCatcherBase(
                grid_size=size_list[i],
                nb_agents_hv=nb_agents_hv_list[i],
                nb_agents_diag=nb_agents_diag_list[i],
                nb_targets=nb_targets_list[i],
                nb_mobiles=nb_mobiles_list[i],
                fow_agents_hv=fow_agents_hv_list[i],
                fow_agents_diag=fow_agents_diag_list[i],
                seed=7,
            )
            for i in range(len(size_list))
        ]
        for i in range(len(envs)):
            envs[i].reset()
            done = False
            lim = 0
            while not (done or lim<500):
                self.check_position_unicity(env[i].world)
                action = env[i].action_space.sample()
                _, _, _, _ = env.step(action)
                lim +=1
    def test_reward(self):
        size_list = [10, 20, 100, 200, 1000, 2000, 3000,30,7]
        nb_agents_hv_list = [10, 4, 10, 20, 100, 200, 1000,24,5]
        nb_agents_diag_list = [10, 4, 10, 20, 100, 200, 1000,10,5]
        nb_targets_list = [5, 4, 10, 20, 100, 200, 1000,5,3]
        nb_mobiles_list = [5, 4, 10, 20, 100, 200, 1000,0,3]
        fow_agents_hv_list = [0,1,2,3,4,10,15,20,25,0,0]
        fow_agents_diag_list = [0,1,2,3,4,10,15,20,25,0,0]
        envs = [
            TeamCatcherBase(
                grid_size=size_list[i],
                nb_agents_hv=nb_agents_hv_list[i],
                nb_agents_diag=nb_agents_diag_list[i],
                nb_targets=nb_targets_list[i],
                nb_mobiles=nb_mobiles_list[i],
                fow_agents_hv=fow_agents_hv_list[i],
                fow_agents_diag=fow_agents_diag_list[i],
                seed=7,
            )
            for i in range(len(size_list))
        ]
        for i in range(len(envs)):
            envs[i].reset()
            done = False
            lim = 0
            score = 0
            while not (done or lim<500):
                prev_targets = envs[i].world.capturedTargets
                prev_mobiles = envs[i].world.capturedMobiles
                action = env[i].action_space.sample()
                _, reward, _, _ = env.step(action)
                score+=reward
                var_targets = envs[i].world.capturedTargets - prev_targets
                var_mobiles = envs[i].world.capturedMobiles - prev_mobiles
                assert reward ==  (var_targets + var_mobiles)
                lim +=1
            assert int(done)*score == int(done)*(envs[i].world.nb_targets + 2*envs[i].world.nb_targets)


        


# if __name__ == "__main__":
#     test = TestGame()
#     test.test_attribute()
#     test.test_reset()
