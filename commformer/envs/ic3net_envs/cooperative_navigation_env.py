import numpy as np
from gym import spaces
import gym


class CN_Env(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self, args):
        self.__version__ = "0.0.1"

        # 添加最大步数限制
        self.max_steps = args.max_steps  # 需要在args中添加该参数
        self.current_step = 0  # 在reset中重置

        # TODO: better config handling
        self.OUTSIDE_CLASS = 1
        self.TIMESTEP_PENALTY = -0.01
        self.episode_over = False

        self.args = args
        self.num_agents = args.num_agents
        self.num_landmarks = args.num_agents
        self.fixed_world_size = 2.0  # 使用固定的大小

        self.multi_agent_init(self.args)

    def multi_agent_init(self, args):
        self.agents = np.zeros((self.num_agents, 2))
        self.landmarks = np.zeros((self.num_landmarks, 2))
        self.stay = not args.no_stay
        self.n_agents = args.num_agents
        if self.stay:
            self.naction = 5
        else:
            self.naction = 4
        tmp_obs = self.reset()
        self.action_space = [spaces.Discrete(5) for _ in range(self.num_agents)]
        # self.observation_space = [spaces.Box(low=0, high=self.fixed_world_size, shape=(tmp_obs[n].shape[0],), dtype=np.float32) for n in range(self.num_agents)]
        self.observation_space = [spaces.Box(
            low=-np.inf,  # 允许负值（相对位置）
            high=np.inf,
            shape=(tmp_obs[n].shape[0],),
            dtype=np.float32
            ) for n in range(self.num_agents)] 
        share_obs_dim = tmp_obs[0].shape[0] * self.n_agents
        self.share_observation_space = [spaces.Box(low=0, high=1, shape=(share_obs_dim, ), dtype=np.float32)
                                  for n in range(self.n_agents)]
        return

    def reset(self):
        self.agents = np.random.rand(self.num_agents, 2) * self.fixed_world_size  # 在固定大小范围内随机生成智能体位置
        self.landmarks = np.random.rand(self.num_landmarks, 2) * self.fixed_world_size  # 在固定大小范围内随机生成目标位置
        self.episode_over = False
        return self._get_obs()

    def step(self, actions):
        self.current_step += 1
        self.episode_over = self.current_step >= self.max_steps

        for i, action in enumerate(actions):
            self._take_action(i, action)
        obs = self._get_obs()
        reward = self._get_reward()
        # 添加成功条件判断（占领所有地标）
        occupied = 0
        for landmark in self.landmarks:
            if any(np.linalg.norm(agent - landmark) < 0.05 for agent in self.agents):
                occupied += 1
        if occupied == self.num_landmarks:
            self.episode_over = True

        done = [self.episode_over] * self.num_agents
        info = {}

        return obs, reward, done, info

    def _take_action(self, idx, action):
        if isinstance(action, np.ndarray):
            assert sum(action) == 1
            action = list(action).index(1)
        # 减小移动步长
        step_size = self.fixed_world_size/10  # 原1.0步幅过大
        if action == 0:  # stay
            pass
        elif action == 1:  # up
            self.agents[idx][1] += step_size
        elif action == 2:  # right 
            self.agents[idx][0] += step_size
        elif action == 3:  # down
            self.agents[idx][1] -= step_size
        elif action == 4:  # left
            self.agents[idx][0] -= step_size
        self.agents[idx] = np.clip(self.agents[idx], 0, self.fixed_world_size)  # 确保智能体不会移动到环境边界之外

    def _get_obs(self):
        obs = []
        for idx, agent in enumerate(self.agents):
            # 包含所有地标的相对位置
            entity_pos = [landmark - agent for landmark in self.landmarks]
            # 包含其他智能体的相对位置
            other_pos = [other - agent for i, other in enumerate(self.agents) if i != idx]
            # 包含自身速度信息（需在类中添加状态记录）
            obs.append(np.concatenate([
                agent,
                *entity_pos,
                *other_pos
            ]).flatten())
        return np.array(obs)

    def _get_reward(self):
        global_rew = 0  # 初始化奖励数组
        # 计算全局奖励
        for landmark in self.landmarks:
            # 找到离地标最近的agent距离
            min_dist = min(np.linalg.norm(agent - landmark) for agent in self.agents)
            global_rew -= min_dist
        
        # 碰撞惩罚
        collision_penalty = 0
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                if np.linalg.norm(self.agents[i] - self.agents[j]) < 0.1:  # 碰撞阈值
                    collision_penalty -= 1
                    
        # 时间步惩罚
        time_penalty = self.TIMESTEP_PENALTY
        
        # 总奖励 = 全局奖励 + 碰撞惩罚 + 时间惩罚 
        total_reward = global_rew + collision_penalty + time_penalty
        
        # 平均分配给各agent（协作场景）
        reward = np.full((self.num_agents, 1), total_reward / self.num_agents)
        return reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass
