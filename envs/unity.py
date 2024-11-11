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
    def __init__(self, action_repeat=4, size=(84, 60), gray=True, noops=0, seed=None):
        # Unity環境の設定
        self.channel = EngineConfigurationChannel()
        print(f"Unity Environment:")
        # ランダムに使用可能なポートを取得
        base_port = get_available_port()

        # 使用可能なポートでUnityEnvironmentを起動
        self.env = UnityEnvironment(file_name="UnityBuild3", base_port=base_port, side_channels=[self.channel])

        self.env.reset()
        print("Reset environment")

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
            "image" : spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        })

    @property
    def action_space(self):
        # 連続的なアクション空間
        return spaces.Box(low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)


    def step(self, action):
        print(action)

        total_reward = 0.0
        for repeat in range(self._action_repeat):
            print("start step")
            action_tuple = ActionTuple(continuous=np.array([action]))  # Unityに渡す形式
            self.env.set_actions(self._behavior_name, action_tuple)
            print(f"set action: {action_tuple}")
            self.env.step()  # 環境を進める
            print("step")
            decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)
            print(f"reward: {decision_steps.reward}")
            total_reward += decision_steps.reward[0]  # 報酬の合計を加算
            print(f"total_reward: {total_reward}\n")
            print(f"position: {decision_steps.obs[1]}")
            print(f"image: {decision_steps.obs[0]}")
            print("finish step")
            if len(terminal_steps) > 0:  # もしエピソードが終了したら
                print(terminal_steps)
                # 前から1つ目の値をposition_x、2つ目の値をposition_zとして格納
                position_x = decision_steps.obs[1][0,0]  # 前から1つ目の値
                position_z = decision_steps.obs[1][0,1]  # 前から2つ目の値
                image= decision_steps.obs[0]
                print(f"distance: {decision_steps.obs[1][0,2]}")
                
                # position_xとposition_zを返す（辞書形式で）
                return {"position_x": position_x, "position_z": position_z, "image": image}, total_reward, True , {}

        # もしエピソードが終了しなければ、位置情報を辞書として返す
        position_x = decision_steps.obs[1][0,0]  # 前から1つ目の値
        position_z = decision_steps.obs[1][0,1]  # 前から2つ目の値
        image= decision_steps.obs[0]
        

        return {"position_x": position_x, "position_z": position_z, "image": image}, total_reward, False, {}


    def reset(self):
        print("before reset")
        self.env.reset()  # 環境をリセット
        print(f"reset")
        decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)
        
        if len(decision_steps) > 0:
            # 初期の観測データを取得
            initial_observation = decision_steps.obs[1]
            print(initial_observation)
            # 前から1つ目の値をposition_x、2つ目の値をposition_zとして格納
            position_x = initial_observation[0,0]  # 前から1つ目の値
            position_z = initial_observation[0,1]  # 前から2つ目の値
            # 画像の観測データを取得
            image = decision_steps.obs[1]         
            return {"position_x": position_x, "position_z": position_z, "image": image}


    def close(self):
        self.env.close()
