from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from shimmy import GymV21CompatibilityV0
import matplotlib.pyplot as plt
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_training_data(metrics, save_path="graphs/training_graph.png"):
    """
    metrics: list of dicts, each describing one line to plot, e.g.
        {"data": rewards, "title": "Total Reward per Episode",
         "ylabel": "Reward Score", "color": "green"}
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))

    # regular list with axes as the only item because function above returns single Axes object if n ==1
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        ax.plot(m["data"], color=m.get("color", "blue"), alpha=0.6)
        ax.set_title(m["title"])
        ax.set_xlabel(m.get("xlabel", "Episodes"))
        ax.set_ylabel(m["ylabel"])
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def get_env(full_color: bool):
    # gym_super_mario_bros only registers with the OLD `gym` package, not
    # gymnasium — so we build it with the legacy gym API first
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # apply preprocessing pipeline
    if not full_color:
        env = GrayScaleObservation(env, keep_dim=False)  # Shape: (240, 256)
    env = ResizeObservation(env, shape=84)  # Shape: (84, 84)
    env = FrameStack(env, num_stack=4)

    # then wrap it so it behaves like a gymnasium env
    env = GymV21CompatibilityV0(env=env)

    return env

def save_checkpoint():
    pass

#translates environment obs to tensors because env outputs lazyframes objects.
def obs_to_tensor(obs):
    arr = np.array(obs, dtype=np.float32)          # LazyFrames -> ndarray
    if arr.ndim == 4 and arr.shape[-1] == 1:        # squeeze stray channel dim if present
        arr = arr.squeeze(-1)
    return torch.tensor(arr).unsqueeze(0).to(device)          # (4, 84, 84) -> (1, 4, 84, 84)



