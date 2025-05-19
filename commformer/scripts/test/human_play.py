from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app
import pygame
import numpy as np


# 定义人类玩家操作类
class HumanAgent:
    def __init__(self):
        self.action_spec = None
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None

    def setup(self, obs_spec, action_spec):
        self.action_spec = action_spec
        self.obs_spec = obs_spec

    def reset(self):
        self.episodes += 1

    def step(self, obs):
        self.steps += 1
        # 在此处理键盘输入
        return self._parse_human_action()  # 返回动作函数

    def _parse_human_action(self):
        """将键盘事件转换为游戏动作"""
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:  # 按下A键攻击
                    return actions.FUNCTIONS.Attack_screen("now", (32, 32))
                elif event.key == pygame.K_m:  # 按下M键移动
                    return actions.FUNCTIONS.Move_screen("now", (64, 64))
        return actions.FUNCTIONS.no_op()  # 默认无操作


# 主函数
def main(unused_argv):
    pygame.init()  # 初始化Pygame
    with sc2_env.SC2Env(
            map_name="DefeatZerglingsAndBanelings",  # 使用简单64地图
            players=[sc2_env.Agent(sc2_env.Race.terran),  # 人类玩家
                     sc2_env.Bot(sc2_env.Race.zerg,  # 内置AI
                                 sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True),  # 启用单位特征
            step_mul=8,  # 降低游戏速度
            realtime=True,  # 关键！启用实时模式
            visualize=True  # 显示游戏画面
    ) as env:
        agent = HumanAgent()
        agent.setup(env.observation_spec(), env.action_spec())

        timestep = env.reset()
        agent.reset()

        while True:
            # 人类玩家动作
            human_action = agent.step(timestep[0])
            # AI动作（此处可替换为模型预测）
            ai_action = [actions.FUNCTIONS.no_op()]
            # 组合动作并执行
            timestep = env.step([human_action, ai_action])

            if timestep[0].last():  # 回合结束
                print("游戏结束！胜利" if timestep[0].reward > 0 else "失败")
                break


if __name__ == "__main__":
    app.run(main)