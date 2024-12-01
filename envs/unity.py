import numpy as np
import torch
import socket
from gym import spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os
import cv2


def get_available_port(start_port=5005, end_port=5100):
    """指定された範囲内で使用可能なポートを取得する"""
    while True:
        port = np.random.randint(start_port, end_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:  # ポートが使用されていない場合
                return port
            
class UnityEnv:
    def __init__(self, action_repeat=2, size=(64, 64), gray=True, noops=0, seed=None):
        # Unity環境の設定
        self.channel = EngineConfigurationChannel()
        print(f"Unity Environment:")
        # 解像度や品質レベルの設定
        self.channel.set_configuration_parameters(width=640, height=480, quality_level=1)

        # ランダムに使用可能なポートを取得
        base_port = get_available_port()
        print(f"Base port: {base_port}")

        # 使用可能なポートでUnityEnvironmentを起動
        self.env = UnityEnvironment(file_name="UnityBuild5", base_port=base_port, side_channels=[self.channel])

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
        if(len(self.spec.observation_specs) > 1):
            self._state_dim = self.spec.observation_specs[0].shape[0]
        self._action_dim = self.spec.action_spec.continuous_size

    @property
    def observation_space(self):
        return spaces.Dict({
            # 画像データ (通常のカメラ画像)
            "image": spaces.Box(0, 255, self._size + (3,), dtype=np.float32),
            
            # 相対座標 (例: x, y の2次元座標)
            "relative_position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            
            # 自分の向き (例: -π ~ π の角度)
            "orientation": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            
            # # 深度カメラからの情報 (深度マップ画像データ)
            # "depth_camera": spaces.Box(low=0, high=np.inf, shape=self._depth_camera_shape, dtype=np.float32)
        })

    @property
    def action_space(self):
        # 連続的なアクション空間
        return spaces.Box(low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)

def step(self, action):
    retries = 3  # リトライ回数の上限
    for attempt in range(retries):
        try:
            # 初回のアクションをUnity形式に変換
            action_tuple = ActionTuple(continuous=np.array([action]))
            self.env.set_actions(self._behavior_name, action_tuple)
            self.env.step()
            # 成功した場合はリトライループを抜ける
            break
        except Exception as e:
            print(f"Step failed on attempt {attempt + 1} with error: {e}")
            if attempt == retries - 1:
                raise

    total_reward = 0.0
    is_terminate = False
    last_decision_steps = None  # 最後の観測データを格納する変数
    for repeat in range(self._action_repeat):
        # アクション設定と進行
        action_tuple = ActionTuple(continuous=np.array([action]))
        self.env.set_actions(self._behavior_name, action_tuple)
        self.env.step()

        # 観測データを取得
        decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)

        # 報酬を累積
        total_reward += decision_steps.reward[0]

        # エピソード終了判定
        if len(terminal_steps) > 0:
            is_terminate = True
            break

        # 最後の観測データを更新
        last_decision_steps = decision_steps

    # 最後の観測データが存在する場合のみ処理
    if last_decision_steps is not None:
        relative_position = np.array([
            last_decision_steps.obs[1][0, 0],  # x座標
            last_decision_steps.obs[1][0, 1]   # z座標
        ]) if len(last_decision_steps.obs) > 1 else np.array([None, None])

        orientation = np.array([
            last_decision_steps.obs[1][0, 2]   # 向き (例: ラジアン値)
        ]) if len(last_decision_steps.obs) > 1 else np.array([None])

        # 画像データ処理
        image = np.array(last_decision_steps.obs[0], dtype=np.float) if len(last_decision_steps.obs) > 0 else None
        if image is not None:
            image = np.squeeze(image, axis=0)  # (1, 60, 84, 3) -> (60, 84, 3)
            image = np.clip(image * 255, 0, 255)  # 0〜255にスケーリング
            image = cv2.resize(image, self._size)
    else:
        # デフォルト値を設定
        relative_position = np.array([None, None])
        orientation = np.array([None])
        image = None

    # 観測データの構築
    obs = {
        "image": image,
        "relative_position": relative_position,
        "orientation": orientation,
        "is_first": False,
        "is_terminal": is_terminate,
    }

    return obs, total_reward, is_terminate, {}


def reset(self):
    self.env.reset()
    decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)

    if len(decision_steps.obs) > 1:
        initial_observation = decision_steps.obs[1]
        relative_position = np.array([
            initial_observation[0, 0],  # x座標
            initial_observation[0, 1]   # z座標
        ])
        orientation = np.array([
            initial_observation[0, 2]   # 向き (例: ラジアン値)
        ])
    else:
        relative_position = np.array([None, None])
        orientation = np.array([None])

    image = np.array(decision_steps.obs[0], dtype=np.float) if len(decision_steps.obs) > 0 else None
    if image is not None:
        image = np.squeeze(image, axis=0)  # (1, 60, 84, 3) -> (60, 84, 3)
        image = np.clip(image * 255, 0, 255)  # 0〜255にスケーリング
        image = cv2.resize(image, self._size)

    obs = {
        "image": image,
        "relative_position": relative_position,
        "orientation": orientation,
        "is_first": True,
        "is_terminal": False,
    }

    return obs