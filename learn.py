from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def display_camera_observation(observation):
    """カメラ観測データを表示する関数"""
    # 観測データの形状を確認
    print(f"Camera observation shape: {observation.shape}")
    
    # グレースケールとカラーで処理を分ける
    if len(observation.shape) == 2:  # グレースケール
        plt.imshow(observation, cmap='gray')
    else:  # カラー
        # Unity からの観測データは [H, W, C] 形式
        plt.imshow(observation)
    
    plt.axis('off')
    plt.show()

def train_agv():
    # Unity環境の設定
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name="UnityBuild2.x86_64", side_channels=[channel])
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
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        step_count = 0
        while True:
            if len(decision_steps) > 0:
                print(decision_steps)
                # ベクター観測とカメラ観測の取得
                vector_obs = decision_steps.obs[1]  # ベクター観測
                camera_obs = decision_steps.obs[0]  # カメラ観測
                
                # 10ステップごとにカメラ画像を表示
                if step_count % 10 == 0:
                    print(f"\nStep {step_count}")
                    print(f"Vector observation shape: {vector_obs.shape}")
                    print(f"Camera observation shape: {camera_obs.shape}")
                    
                    # カメラ画像の表示
                    display_camera_observation(camera_obs[0])  # 最初のエージェントの観測のみ表示
                
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
                    display_camera_observation(terminal_steps.obs[1][0])
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