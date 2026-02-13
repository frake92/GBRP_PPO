import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import os

# --- BEÁLLÍTÁSOK ---
MODEL_PATH = "runs/Seed42/BipedalWalkerGBRP42_2/final_model.zip"
VIDEO_FOLDER = "runs/Seed42/BipedalWalkerGBRP42_2" # Ide menti majd a videókat

def main():
    # 1. Alap környezet (rgb_array mód kell a videóhoz)
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

    # 2. Videó mentő wrapper hozzáadása
    # Ez minden epizódot felvesz (episode_trigger=lambda x: True)
    env = RecordVideo(env, video_folder=VIDEO_FOLDER, 
                      episode_trigger=lambda x: True, 
                      name_prefix="gbrp_walker_video")

    # 3. Modell betöltése
    try:
        model = PPO.load(MODEL_PATH)
        print(f"Modell betöltve innen: {MODEL_PATH}")
    except Exception as e:
        print(f"Hiba a betöltésnél: {e}")
        return

    obs, _ = env.reset()
    print("Felvétel indul... fuss le legalább egy teljes kört!")

    # Futtatunk pár epizódot
    for episode in range(3): # Itt megadhatod hány kört vegyen fel
        terminated = False
        truncated = False
        obs, _ = env.reset()
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Megjelenítés a képernyőn is (opcionális, de jó látni)
            env.render() 
        
        print(f"{episode+1}. epizód kész, mentve!")

    # Fontos a bezárás, különben nem menti el rendesen a fájlt!
    env.close()
    print(f"A videókat itt találod: {os.path.abspath(VIDEO_FOLDER)}")

if __name__ == "__main__":
    main()