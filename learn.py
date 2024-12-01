from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import numpy as np
import torch
# import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt

def display_camera_observation(observation):
    """カメラ観測データを表示する関数"""
    print(f"Camera observation shape: {observation.shape}")
    
    # 画像のスケーリング: [0, 1] → [0, 255]
    observation = np.clip(observation * 255, 0, 255).astype(np.uint8)
    
    if len(observation.shape) == 2:  # グレースケール
        plt.imshow(observation, cmap='gray')
    else:  # カラー
        plt.imshow(observation)
    
    plt.axis('off')  # 軸を非表示にする
    plt.show()  # 画像を表示


def train_agv():
    # Unity環境の設定
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=None, side_channels=[channel])
    env.reset()
    
    # 環境の設定
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    print(f"Behavior Name: {behavior_name}")
    
    # 観測仕様の確認
    observation_specs = spec.observation_specs
    print("\nObservation Specifications:")
    for i, obs_spec in enumerate(observation_specs):
        print(f"Observation {i}:")
        print(f"  Shape: {obs_spec.shape}")
        print(f"  Type: {obs_spec.observation_type}")
        print(f"  Name: {obs_spec.name}")
    
    # 状態次元と行動次元の取得
    vector_obs_dim = spec.observation_specs[0].shape[0]  # ベクター観測の次元
    action_dim = spec.action_spec.continuous_size
    print(f"\nVector Observation Dimension: {vector_obs_dim}")
    print(f"Action Dimension: {action_dim}")
    
    for episode in range(10):
        print(f"\nStarting episode {episode + 1}...")
        env.reset()
        
        step_count = 0
        while step_count < 1000:
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if len(decision_steps) > 0: 
                print(f"decision_steps: {decision_steps}")

                # カメラ画像を行列として取得（浮動小数点型のまま）
                image = np.array(decision_steps.obs[0], dtype=np.float32) if len(decision_steps.obs) > 0 else None

                if image is not None:
                    image = np.squeeze(image, axis=0)  # (1, 60, 84, 3) -> (60, 84, 3)
                    
                    # 画像のスケーリング: [0, 1] → [0, 255]（整数型に変換せず表示）
                    # image = np.clip(image * 255, 0, 255)  # 0〜255にスケーリング

                    # 画像の表示
                    display_camera_observation(image)

                        
                    # 行列の詳細を出力
                    print("Camera Image Matrix:")
                    print(image)
                    
                    # 行列の形状と詳細情報
                    print("\nMatrix Details:")
                    print(f"Shape: {image.shape}")
                    print(f"Data Type: {image.dtype}")
                    print(f"Min Value: {image.min()}")
                    print(f"Max Value: {image.max()}")

                    print(decision_steps.obs[1])

                    relative_position = (decision_steps.obs[1][0][0], decision_steps.obs[1][0][1])

                    relative_angle = decision_steps.obs[1][0][2]       # 相対角度

                    # 出力
                    print(f"  Relative Position: {relative_position}")
                    print(f"  Relative Angle: {relative_angle}")

                
                # ランダムな行動の生成と実行
                random_action = np.random.uniform(low=-1.0, high=1.0, size=(1, action_dim))
                action_tuple = ActionTuple(continuous=random_action)
                env.set_actions(behavior_name, action_tuple)
                
                # 環境を1ステップ進める
                env.step()
                
                # 次の状態の取得
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                step_count += 1
            
            # エピソード終了判定
            if len(terminal_steps) > 0:
                print(f"Episode {episode + 1} completed after {step_count} steps")
                # 終了時のカメラ画像も表示
                if len(terminal_steps.obs) > 1:  # カメラ観測が存在する場合
                    print("Terminal observation:")
                    # display_camera_observation(terminal_steps.obs[1][0])
                break
    
    env.close()

def save_observation_as_image(observation, filename):
    """観測データを画像として保存する関数"""
    if len(observation.shape) == 2:  # グレースケール
        img = Image.fromarray((observation * 255).astype(np.uint8))
    else:  # カラー
        img = Image.fromarray((observation * 255).astype(np.uint8))
    
    img.save(filename)
    print(f"Saved image to {filename}")

if __name__ == "__main__":
    train_agv()
