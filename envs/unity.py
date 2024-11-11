import numpy as np
import torch
import socket
from gym import spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os


def get_available_port(start_port=5005, end_port=5100):
    """指定された範囲内で使用可能なポートを取得する"""
    while True:
        port = np.random.randint(start_port, end_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:  # ポートが使用されていない場合
                return port
            
class UnityEnv:
    def __init__(self, action_repeat=4, size=(84, 84), gray=True, noops=0, seed=None):
        # Unity環境の設定
        self.channel = EngineConfigurationChannel()
        print(f"Unity Environment:")
        # ランダムに使用可能なポートを取得
        base_port = get_available_port()

        # 使用可能なポートでUnityEnvironmentを起動
        self.env = UnityEnvironment(file_name="UnityBuild1", base_port=base_port, side_channels=[self.channel])
        

        self.env.reset()

        self._action_repeat = action_repeat
        self._size = size
        self._gray = gray
        self._noops = noops
        self._seed = seed
        self._random = np.random.RandomState(seed)

        # 環境の設定
        behavior_name = list(self.env.behavior_specs)[0]
        self.spec = self.env.behavior_specs[behavior_name]
        self._behavior_name = behavior_name
        self._state_dim = self.spec.observation_specs[0].shape[0]
        self._action_dim = self.spec.action_spec.continuous_size
        
    @property
    def observation_space(self):
        return spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32),
            # 他の観測データがある場合は追加
            # "another_obs": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
        })

    @property
    def action_space(self):
        # 連続的なアクション空間
        return spaces.Box(low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)

    def step(self, action):
        # actionがリストであれば、最初に最大値を選択
        if len(action.shape) >= 1:
            action = np.argmax(action)
        
        total_reward = 0.0
        for repeat in range(self._action_repeat):
            action_tuple = ActionTuple(continuous=np.array([action]))  # Unityに渡す形式
            self.env.set_actions(self._behavior_name, action_tuple)

            self.env.step()  # 環境を進める
            decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)

            total_reward += decision_steps.reward[0]  # 報酬の合計を加算

            if len(terminal_steps) > 0:  # もしエピソードが終了したら
                return np.array(decision_steps.obs[0], total_reward, True)

        return np.array(decision_steps.obs[0], total_reward, False)

    def reset(self):
        self.env.reset()
        
        decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)
       
        if len(decision_steps) > 0:
            initial_observation = decision_steps.obs[0]
            print(f"Reset: {initial_observation}")
            return np.array([initial_observation, 0.0, True])

    def close(self):
        self.env.close()
