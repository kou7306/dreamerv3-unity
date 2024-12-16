import numpy as np
import torch
import socket
import time  # 追加：一時停止に使用
from gym import spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os
import cv2


used_ports = set()  # 使用済みのポートを管理するためのセット

def get_available_port(start_port=5005, end_port=5100):
    """指定された範囲内で使用可能なポートを取得し、使用済みポートセットに登録する"""
    while True:
        port = np.random.randint(start_port, end_port)
        if port in used_ports:
            continue  # すでに使用されているポートはスキップ
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:  # ポートが使用されていない場合
                used_ports.add(port)  # 使用済みポートに追加
                return port


class UnityEnv:
    def __init__(self, action_repeat=10, size=(64, 64), gray=True, noops=0, seed=None, retries=3, startup_delay=10):
        """Unity環境の初期化"""
        self.retries = retries  # リトライ回数の指定
        self.startup_delay = startup_delay  # 環境起動後の待機時間
        self._init_env(action_repeat, size, gray, noops, seed)

    def _init_env(self, action_repeat, size, gray, noops, seed):
        """Unity 環境を初期化する。失敗した場合はリトライする。"""
        for attempt in range(self.retries):
            try:
                # Unity 環境の設定
                self.channel = EngineConfigurationChannel()
                print(f"Attempt {attempt + 1}: Initializing Unity Environment...")
                
                # 解像度や品質レベルの設定
                self.channel.set_configuration_parameters(width=480, height=320, quality_level=1)

                # ランダムに使用可能なポートを取得
                base_port = get_available_port()
                print(f"Base port: {base_port}")

                # Unity 環境の初期化
                self.env = UnityEnvironment(file_name="UnityBuild7", base_port=base_port, side_channels=[self.channel], timeout_wait=60)
                self.env.reset()
                print("Environment successfully initialized.")

                # **環境起動後に待機**
                print(f"Waiting for {self.startup_delay} seconds to ensure environment is fully initialized...")
                time.sleep(self.startup_delay)  # 待機時間を追加

                self._action_repeat = action_repeat
                self._size = size
                self._gray = gray
                self._noops = noops
                self._seed = seed
                self._random = np.random.RandomState(seed)
                
                behavior_name = list(self.env.behavior_specs)[0]
                self.spec = self.env.behavior_specs[behavior_name]
                self._behavior_name = behavior_name
                
                if len(self.spec.observation_specs) > 1:
                    self._state_dim = self.spec.observation_specs[0].shape[0]
                
                self._action_dim = self.spec.action_spec.continuous_size
                self.lidar_dim = 3840  # Lidarのデータ次元数
                break  # 初期化に成功したらループを抜ける
            except Exception as e:
                print(f"Initialization failed on attempt {attempt + 1} with error: {e}")
                if attempt == self.retries - 1:
                    raise  # リトライの最後の試行でも失敗した場合は例外を投げる

    @property
    def observation_space(self):
        return spaces.Dict({
            "image": spaces.Box(0, 255, self._size + (3,), dtype=np.float32),
            "relative_position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "orientation": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "lidar": spaces.Box(low=0, high=np.inf, shape=(self.lidar_dim,), dtype=np.float32),
        })

    @property
    def action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)

    def step(self, action):
        total_reward = 0.
        is_terminate = False
        last_decision_steps = None
        print("action set",action)
        for repeat in range(self._action_repeat):
            action_tuple = ActionTuple(continuous=np.array([action]))
            self.env.set_actions(self._behavior_name, action_tuple)
            self.env.step()
            decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)
            
            if len(terminal_steps) > 0:
                is_terminate = True
                print("Episode terminated")
                break
            
            total_reward += decision_steps.reward[0]
            print(f"Step {repeat + 1}: Reward {decision_steps.reward[0]}")
            last_decision_steps = decision_steps
            break  # 正常にステップが完了したらリトライループを抜ける
        print(f"Total Reward: {total_reward}")
        obs = self._create_observation(last_decision_steps, is_terminate)
        return obs, total_reward, is_terminate, {}

    def _create_observation(self, decision_steps, is_terminate):
        if decision_steps is not None:
            relative_position = np.array([decision_steps.obs[1][0, 0], decision_steps.obs[1][0, 1]])
            orientation = np.array([decision_steps.obs[1][0, 2]])
            image = np.array(decision_steps.obs[0], dtype=np.float32)
            image = np.squeeze(image, axis=0)
            image = np.clip(image * 255, 0, 255)
            image = cv2.resize(image, self._size)
            lidar_data = np.array(decision_steps.obs[1][0, 3:], dtype=np.float32)
        else:
            relative_position = np.zeros((2,))
            orientation = np.zeros((1,))
            image = np.zeros((self._size[0], self._size[1], 3))
            lidar_data = np.zeros(self.lidar_dim)

        print(f"relative_position: {relative_position}")
        print(f"orientation: {orientation}")
        print(f"lidar_data: {lidar_data}, length: {len(lidar_data)}")
        
        obs = {
            "image": image,
            "relative_position": relative_position,
            "orientation": orientation,
            "lidar": lidar_data,
            "is_first": False,
            "is_terminal": is_terminate,
        }
        return obs

    def reset(self):
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self._behavior_name)
        obs = self._create_observation(decision_steps, is_terminate=False)
        obs["is_first"] = True
        return obs

    def close(self):
        self.env.close()
        print("Unity environment closed.")
