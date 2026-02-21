import numpy as np
from collections import defaultdict

# -------------------------------------------------
# Reward Metrics Buffer
# -------------------------------------------------

class RewardMetricsBuffer:
    def __init__(self):
        self.data = defaultdict(list)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(float(v))

    def stats(self):
        stats = {}
        for k, vals in self.data.items():
            arr = np.array(vals, dtype=np.float32)
            stats[k] = {
                "mean": arr.mean(),
                "std": arr.std() + 1e-8,
                "rms": np.sqrt(np.mean(arr ** 2))
            }
        return stats

    def clear(self):
        self.data.clear()


# -------------------------------------------------
# Auto weight update rule
# -------------------------------------------------

def update_weight(w, rms, target, lr=0.05, w_min=0.01, w_max=2.0):
    ratio = target / (rms + 1e-8)
    ratio = np.clip(ratio, 0.5, 2.0)
    new_w = w * (ratio ** lr)
    return np.clip(new_w, w_min, w_max)


# -------------------------------------------------
# Simulated environment reward signals
# -------------------------------------------------

def fake_reward_components():
    """
    Simulates raw (unweighted) reward components
    similar to your environment.
    """
    r_goal = np.random.uniform(-0.05, 0.05)

    # steering change signal
    r_steer = -np.random.uniform(0.0, 0.02)

    # obstacle / shark sometimes active
    r_shark = -np.random.uniform(0.0, 0.2) if np.random.rand() < 0.1 else 0.0

    # agent proximity sometimes active
    r_agent = -np.random.uniform(0.0, 0.15) if np.random.rand() < 0.15 else 0.0

    # boundary sometimes active
    r_edge = -np.random.uniform(0.0, 0.2) if np.random.rand() < 0.1 else 0.0

    # alignment almost always active
    r_align = np.random.uniform(-0.3, 0.3)

    return {
        "r_goal": r_goal,
        "r_steer": r_steer,
        "r_shark": r_shark,
        "r_agent": r_agent,
        "r_edge": r_edge,
        "r_align": r_align,
    }


# -------------------------------------------------
# Main simulation loop
# -------------------------------------------------

if __name__ == "__main__":

    n_agents = 4
    total_steps = 20000
    tune_interval = 2000

    # Initial weights (like your shaping weights)
    weights = {
        "r_goal": 1.0,
        "r_steer": 0.1,
        "r_shark": 0.5,
        "r_agent": 0.3,
        "r_edge": 0.5,
        "r_align": 0.3,
    }

    buffer = RewardMetricsBuffer()

    for step in range(1, total_steps + 1):

        # ---- Simulate multi-agent steps ----
        for _ in range(n_agents):
            comps = fake_reward_components()
            buffer.add(**comps)

        # ---- Periodic weight tuning ----
        if step % tune_interval == 0:
            stats = buffer.stats()

            target = stats["r_goal"]["rms"]

            print(f"\nStep {step}")
            print("RMS stats:")
            for k, s in stats.items():
                print(f"{k:8s}  rms={s['rms']:.4f}")

            # Update weights (except goal)
            for k in weights:
                if k != "r_goal":
                    weights[k] = update_weight(
                        weights[k],
                        stats[k]["rms"],
                        target
                    )

            print("\nUpdated weights:")
            for k, w in weights.items():
                print(f"{k:8s}  w={w:.3f}")

            buffer.clear()