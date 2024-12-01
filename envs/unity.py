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
    def __init__(self, action_repeat=4, size=(64, 64), gray=True, noops=0, seed=None):
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
            # "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32),
            # 他の観測データがある場合は追加
            "image": spaces.Box(0, 255, self._size + (3,), dtype=np.float32)
        })

    @property
    def action_space(self):
        # 連続的なアクション空間
        return spaces.Box(low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)

    def step(self, action):
        retries = 3  # リトライ回数の上限
        for attempt in range(retries):
            try:
                # アクションをUnity形式に変換
                action_tuple = ActionTuple(continuous=np.array([action]))
                # アクションをUnity環境に設定
                self.env.set_actions(self._behavior_name, action_tuple)
                # Unity環境を進行
                self.env.step()
                # ステップの結果を取得
                decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)

                if len(decision_steps.obs) > 1:
                    # 配列としてx座標とz座標を取得
                    position_x = np.array([decision_steps.obs[1][0, 0]])
                    position_z = np.array([decision_steps.obs[1][0, 1]])
                else:
                    position_x, position_z = np.array([None]), np.array([None])

                # 成功した場合はループを抜けて後続処理を実行
                break
            except Exception as e:
                print(f"Step failed on attempt {attempt + 1} with error: {e}")
                # リトライ回数が上限に達した場合はエラーを発生
                if attempt == retries - 1:
                    raise  # リトライ限度を超えた場合、エラーを発生して停止

        # 成功した場合の後続処理
        total_reward = 0.0
        is_terminate = False
        for repeat in range(self._action_repeat):
            action_tuple = ActionTuple(continuous=np.array([action]))
            self.env.set_actions(self._behavior_name, action_tuple)
            self.env.step()

            decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)
            total_reward += decision_steps.reward[0]  # 報酬の合計を加算
            print("reward: ", decision_steps.reward[0])

            if len(terminal_steps) > 0:  # エピソードが終了した場合
                is_terminate = True
                break

        # 観測データの処理
        image = np.array(decision_steps.obs[0], dtype=np.float) if len(decision_steps.obs) > 0 else None
        if image is not None:
            image = np.squeeze(image, axis=0)  # (1, 60, 84, 3) -> (60, 84, 3)
            image = np.clip(image * 255, 0, 255)  # 0〜255にスケーリング
            image = cv2.resize(image, self._size)


        print("image", image)
        obs = {
            "image": image,
            "is_first": False,  # 初回ではないためFalse
            "is_terminal": is_terminate,  # エピソード終了かどうか
        }

        print("Observation details:")
        for key, val in obs.items():
            print(f"Key: {key}, Shape: {np.shape(val)}, Type: {type(val)}")

        return obs, total_reward, is_terminate, {}


    def reset(self):
        print("before reset")
        self.env.reset()  # 環境をリセット
        print("reset")
        decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)

        if len(decision_steps.obs) > 1:
            # 初期の観測データを取得
            initial_observation = decision_steps.obs[1]
            print(initial_observation)
            # 配列としてx座標とz座標を取得
            position_x = np.array([initial_observation[0, 0]])
            position_z = np.array([initial_observation[0, 1]])
        else:
            position_x, position_z = np.array([None]), np.array([None])

        # 観測データの処理
        image = np.array(decision_steps.obs[0], dtype=np.float) if len(decision_steps.obs) > 0 else None
        if image is not None:
            image = np.squeeze(image, axis=0)  # (1, 60, 84, 3) -> (60, 84, 3)
            image = np.clip(image * 255, 0, 255)  # 0〜255にスケーリング
            image = cv2.resize(image, self._size)


        obs = {
            "image": image,
            "is_first": True,  # 初回のリセット後はTrue
            "is_terminal": False,  # リセット時は終了していない
        }

        print("step_obs", obs)

        return obs


    def close(self):
        self.env.close()
