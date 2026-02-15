import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder


# --- BEÁLLÍTÁSOK ---
ENV_ID = "BipedalWalker-v3"
SEED = 42
MODEL_PATH = "runs/Seed42/BipedalWalker - Modified Gymnasium/BipedalWalkerGBRPMD42/final_model.zip"
VIDEO_FOLDER = "runs/Seed42/BipedalWalker - Modified Gymnasium/BipedalWalkerGBRPMD42/"  # Ide menti majd a videókat


def main():
    # Egyszerű vektorizált env BipedalWalkerhez (ugyanúgy, mint tanításkor a network.py-ben)
    env = make_vec_env(ENV_ID, n_envs=1, seed=SEED)

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
        name_prefix="bipedalwalker_gbrp",
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