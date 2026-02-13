import gymnasium as gym
from stable_baselines3 import PPO
import os
import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from cnn_network import build_env

# --- BEÁLLÍTÁSOK ---
MODEL_PATH = "runs/seaquest_gbrp/final_model.zip"
VIDEO_FOLDER = "runs/seaquest_gbrp" # Ide menti majd a videókat

def main():
    # Ugyanaz a wrapper lánc, mint tanításkor, csak 1 env-vel
    env = build_env("ALE/Seaquest-v5", n_envs=1, seed=0)

    # 3. Modell betöltése
    try:
        model = PPO.load(MODEL_PATH)
        print(f"Modell betöltve innen: {MODEL_PATH}")
    except Exception as e:
        print(f"Hiba a betöltésnél: {e}")
        return

    # 2. Videó mentő wrapper VecEnv-hez
    #    Minden epizódról készít videót (record_video_trigger=lambda x: True)
    env = VecVideoRecorder(
        env,
        VIDEO_FOLDER,
        record_video_trigger=lambda ep_id: True,
        video_length=10_000,
        name_prefix="seaquest_ppo",
    )

    obs = env.reset()
    print("Felvétel indul... pár epizódot rögzítek!")

    # Futtatunk pár epizódot (n_envs=1, ezért a done flag egy 1 elemű tömb)
    n_episodes = 3
    episode = 0
    while episode < n_episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        # Opcionális render (ha van megjelenítő)
        env.render()

        if dones[0]:
            episode += 1
            print(f"{episode}. epizód kész, mentve!")

    # Fontos a bezárás, különben nem menti el rendesen a fájlt!
    env.close()
    print(f"A videókat itt találod: {os.path.abspath(VIDEO_FOLDER)}")

if __name__ == "__main__":
    main()