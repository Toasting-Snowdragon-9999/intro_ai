import os
import random
import argparse
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Globals:
ACTIONS = ("up", "down", "left", "right")

# Rewards, terminals and obstacles are characters:
REWARDS = {" ": -1, ".": -1, "+": -1, "-": -100}
TERMINALS = ("+", "-")  # Note a terminal should also have a reward assigned
OBSTACLES = ("#",)

# Discount factor
gamma = 1

# The probability of a random move:
rand_move_probability = 0


class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Create an empty world where the agent can move to all cells
        self.grid = np.full((width, height), " ", dtype="U1")

    def add_obstacle(self, start_x: int, start_y: int, end_x: Optional[int] = None, end_y: Optional[int] = None) -> None:
        """
        Create an obstacle in either a single cell or rectangle.
        """
        if end_x is None:
            end_x = start_x
        if end_y is None:
            end_y = start_y

        self.grid[start_x : end_x + 1, start_y : end_y + 1] = OBSTACLES[0]

    def add_reward(self, x: int, y: int, reward: str) -> None:
        assert reward in REWARDS, f"{reward} not in {REWARDS}"
        self.grid[x, y] = reward

    def add_terminal(self, x: int, y: int, terminal: str) -> None:
        assert terminal in TERMINALS, f"{terminal} not in {TERMINALS}"
        self.grid[x, y] = terminal

    def is_obstacle(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return self.grid[x, y] in OBSTACLES

    def is_terminal(self, x: int, y: int) -> bool:
        return self.grid[x, y] in TERMINALS

    def get_reward(self, x: int, y: int) -> float:
        """
        Return the reward associated with a given location.
        """
        return REWARDS[self.grid[x, y]]

    def get_next_state(self, current_state: Tuple[int, int], action: str) -> Optional[Tuple[int, int]]:
        """
        Get the next state given a current state and an action. The outcome can be
        stochastic where rand_move_probability determines the probability of
        ignoring the action and performing a random move.
        """
        assert action in ACTIONS, f"Unknown action {action} must be one of {ACTIONS}"

        x, y = current_state

        # If our current state is a terminal, there is no next state
        if self.grid[x, y] in TERMINALS:
            return None

        # Check if a random action should be performed
        if np.random.rand() < rand_move_probability:
            action = np.random.choice(ACTIONS)

        if action == "up":
            y -= 1
        elif action == "down":
            y += 1
        elif action == "left":
            x -= 1
        elif action == "right":
            x += 1

        # If the next state is an obstacle, stay in the current state
        return (x, y) if not self.is_obstacle(x, y) else current_state


class CliffWorld(World):
    def __init__(self):
        super().__init__(width=12, height=4)

        # Start and goal on y=0 row as in existing project convention.
        self.start_state = (0, 0)
        self.goal_state = (11, 0)

        self.add_terminal(*self.goal_state, "+")

        # Cliff cells between start and goal on the same row.
        self.cliff_cells = {(x, 0) for x in range(1, 11)}

    def is_cliff(self, x: int, y: int) -> bool:
        return (x, y) in self.cliff_cells

    def step(self, current_state: Tuple[int, int], action: str) -> Tuple[Optional[Tuple[int, int]], float, bool]:
        """
        Book-exact cliff behavior for the transition event:
        stepping into cliff gives -100 and teleports to start while episode continues.
        """
        proposed_next_state = super().get_next_state(current_state, action)

        if proposed_next_state is None:
            return None, 0.0, True

        x, y = proposed_next_state
        if self.is_cliff(x, y):
            return self.start_state, -100.0, False

        reward = self.get_reward(*proposed_next_state)
        done = self.is_terminal(*proposed_next_state)
        return proposed_next_state, float(reward), done


def _epsilon_greedy_action(q_table: np.ndarray, state: Tuple[int, int], epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return random.randrange(len(ACTIONS))
    return int(np.argmax(q_table[state[0], state[1], :]))


def _env_step(world: World, state: Tuple[int, int], action: str) -> Tuple[Optional[Tuple[int, int]], float, bool]:
    if hasattr(world, "step"):
        next_state, reward, done = world.step(state, action)  # type: ignore[attr-defined]
        return next_state, float(reward), bool(done)

    next_state = world.get_next_state(state, action)
    if next_state is None:
        return None, 0.0, True

    reward = float(world.get_reward(*next_state))
    done = world.is_terminal(*next_state)
    return next_state, reward, done


def _debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def n_step_sarsa(
    world: World,
    start_state: Tuple[int, int],
    n: int,
    Q_table: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    episodes: int = 1000,
    max_steps: int = 1000,
    track_returns: bool = False,
    debug: bool = False,
    debug_episode_interval: int = 25,
    debug_step_interval: int = 200,
):
    if Q_table is None:
        Q_table = np.full((world.width, world.height, len(ACTIONS)), 0.0)

    all_steps: List[int] = []
    all_returns: List[float] = []

    _debug_print(
        debug,
        f"[n_step_sarsa] start n={n}, episodes={episodes}, max_steps={max_steps}, alpha={alpha}, gamma={gamma}, epsilon={epsilon}",
    )

    for episode_idx in range(episodes):
        # Ring buffers (size n+1)
        S: List[Optional[Tuple[int, int]]] = [None] * (n + 1)
        A: List[Optional[int]] = [None] * (n + 1)
        R: List[float] = [0.0] * (n + 1)

        S[0] = start_state
        A[0] = _epsilon_greedy_action(Q_table, start_state, epsilon)

        T = float("inf")
        t = 0
        steps = 0
        episode_return = 0.0

        while True:
            if t < T:
                if steps >= max_steps:
                    # Truncate the episode cleanly; using t would prevent tau from ever reaching T-1.
                    T = t + 1
                    _debug_print(
                        debug,
                        f"[n_step_sarsa] episode={episode_idx + 1}/{episodes} hit max_steps={max_steps} at t={t}",
                    )
                else:
                    s_t = S[t % (n + 1)]
                    a_t = A[t % (n + 1)]
                    assert s_t is not None and a_t is not None

                    next_state, reward, done = _env_step(world, s_t, ACTIONS[a_t])
                    R[(t + 1) % (n + 1)] = reward
                    S[(t + 1) % (n + 1)] = next_state
                    episode_return += reward
                    steps += 1

                    if done:
                        T = t + 1
                        _debug_print(
                            debug,
                            f"[n_step_sarsa] episode={episode_idx + 1}/{episodes} reached terminal at step={steps}, return={episode_return:.3f}",
                        )
                    else:
                        assert next_state is not None
                        A[(t + 1) % (n + 1)] = _epsilon_greedy_action(Q_table, next_state, epsilon)

                if debug and steps > 0 and steps % debug_step_interval == 0:
                    _debug_print(
                        True,
                        f"[n_step_sarsa] heartbeat episode={episode_idx + 1}/{episodes}, steps={steps}, t={t}, T={T}",
                    )

            tau = t - n + 1

            if tau >= 0:
                G = 0.0
                if T == float("inf"):
                    upper = tau + n
                else:
                    upper = min(tau + n, int(T))
                for i in range(tau + 1, upper + 1):
                    G += (gamma ** (i - tau - 1)) * R[i % (n + 1)]

                if tau + n < T:
                    s_tau_n = S[(tau + n) % (n + 1)]
                    a_tau_n = A[(tau + n) % (n + 1)]
                    assert s_tau_n is not None and a_tau_n is not None
                    G += (gamma ** n) * Q_table[s_tau_n[0], s_tau_n[1], a_tau_n]

                s_tau = S[tau % (n + 1)]
                a_tau = A[tau % (n + 1)]
                assert s_tau is not None and a_tau is not None
                Q_table[s_tau[0], s_tau[1], a_tau] += alpha * (G - Q_table[s_tau[0], s_tau[1], a_tau])

            if tau >= T - 1:
                _debug_print(
                    debug,
                    f"[n_step_sarsa] draining finished at episode={episode_idx + 1}/{episodes}, t={t}, tau={tau}, T={T}",
                )
                break

            t += 1

        all_steps.append(steps)
        all_returns.append(episode_return)
        if debug and ((episode_idx + 1) % debug_episode_interval == 0 or episode_idx == episodes - 1):
            _debug_print(
                True,
                f"[n_step_sarsa] completed episode={episode_idx + 1}/{episodes}, steps={steps}, return={episode_return:.3f}",
            )

    _debug_print(debug, f"[n_step_sarsa] done n={n}")
    if track_returns:
        return Q_table, all_steps, all_returns
    return Q_table, all_steps


def q_visualizer(world: World, q: np.ndarray, title: Optional[str] = None) -> np.ndarray:
    new_q = np.full((world.width, world.height), "", dtype=object)
    encoding = sys.stdout.encoding or ""
    unicode_ok = True
    try:
        "↑↓←→".encode(encoding if encoding else "utf-8")
    except UnicodeEncodeError:
        unicode_ok = False

    action_symbols = (
        {"up": "↑", "down": "↓", "left": "←", "right": "→"}
        if unicode_ok
        else {"up": "^", "down": "v", "left": "<", "right": ">"}
    )
    for x in range(world.width):
        for y in range(world.height):
            if isinstance(world, CliffWorld) and world.is_cliff(x, y):
                new_q[x, y] = "C"
            elif world.is_terminal(x, y):
                new_q[x, y] = "+"
            elif world.is_obstacle(x, y):
                new_q[x, y] = "#"
            else:
                new_q[x, y] = action_symbols[ACTIONS[int(np.argmax(q[x, y, :]))]]

    if title is None:
        title = "Best action at each state (arrows show direction):"
    print(title)

    # table = np.flipud(new_q.T)
    table = new_q.T
    x_header = "     " + " ".join(f"{x:>2}" for x in range(world.width))
    print(x_header)
    for row_idx, row in enumerate(table):
        display_row = world.height - row_idx
        print(f"row={display_row:>2} | " + " ".join(f"{str(cell):>2}" for cell in row))

    return new_q


def run_validation_checks(results: Dict[str, object], episodes: int) -> None:
    env = CliffWorld()

    next_state, reward, done = env.step(env.start_state, "right")
    assert next_state == env.start_state, "Cliff step should reset to start state"
    assert reward == -100.0, "Cliff step should return -100 reward"
    assert done is False, "Cliff step should not terminate the episode"

    next_state, reward, done = env.step((10, 0), "right")
    assert next_state == env.goal_state, "Expected transition into goal"
    assert done is True, "Goal state should terminate the episode"
    assert reward == -1.0, "Goal step reward should be -1"

    mean_returns_by_n = results["mean_returns_by_n"]
    assert set(mean_returns_by_n.keys()) == {1, 2, 3, 4}, "Sweep must contain n={1,2,3,4}"

    for n, curve in mean_returns_by_n.items():
        assert curve.shape == (episodes,), f"Mean return curve for n={n} has wrong shape"
        assert np.isfinite(curve).all(), f"Mean return curve for n={n} contains non-finite values"

    plot_path = results["plot_path"]
    assert os.path.exists(plot_path), f"Expected output plot at {plot_path}"

    first_k = max(1, min(100, episodes // 2))
    last_k = max(1, min(100, episodes // 2))
    improved = any(
        mean_returns_by_n[n][-last_k:].mean() > mean_returns_by_n[n][:first_k].mean() for n in mean_returns_by_n
    )
    assert improved, "Expected at least one n to improve from early to late episodes"


def smooth_curve(data, window_size=20):
    if window_size <= 1:
        return data
    w = np.ones(window_size) / window_size
    # 'same' preserves the original length
    return np.convolve(data, w, mode='same')

def run_cliff_nstep_experiment(
    n_values: Iterable[int] = (1, 2, 3, 4),
    episodes: int = 500,
    runs_per_n: int = 1000,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    max_steps: int = 1000,
    seed_base: int = 123,
    out_path: str = "sarsa/artifacts/cliff_nstep_returns.png",
    visualize_policy: bool = True,
    debug: bool = False,
) -> Dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_values = list(n_values)
    mean_returns_by_n: Dict[int, np.ndarray] = {}
    mean_q_by_n: Dict[int, np.ndarray] = {}

    _debug_print(
        debug,
        f"[experiment] start n_values={n_values}, episodes={episodes}, runs_per_n={runs_per_n}, max_steps={max_steps}",
    )

    experiment_start = time.perf_counter()
    for n in n_values:
        n_start = time.perf_counter()
        _debug_print(debug, f"[experiment] start n={n}")
        run_returns = []
        run_q = []
        for run_idx in range(runs_per_n):
            run_start = time.perf_counter()
            _debug_print(debug, f"n={n} run {run_idx + 1}/{runs_per_n}")
            seed = seed_base + (n * 1000) + run_idx
            np.random.seed(seed)
            random.seed(seed)

            env = CliffWorld()
            world = env
            # world = World(12, 8)

            # world.add_terminal(world.width-1, world.height-1, "+")
            # for x in range(1, world.width-1):
            #     world.add_terminal(x, world.height-1, "-")


            q_table, _, episode_returns = n_step_sarsa(
                world=world,
                start_state=env.start_state,
                n=n,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                episodes=episodes,
                max_steps=max_steps,
                track_returns=True,
                debug=debug,
            )
            run_returns.append(np.asarray(episode_returns, dtype=float))
            run_q.append(q_table)
            _debug_print(
                debug,
                f"[experiment] finished n={n} run={run_idx + 1}/{runs_per_n} in {time.perf_counter() - run_start:.2f}s",
            )

        mean_returns_by_n[n] = np.mean(np.vstack(run_returns), axis=0)
        mean_q_by_n[n] = np.mean(np.stack(run_q, axis=0), axis=0)
        _debug_print(debug, f"[experiment] aggregated n={n} in {time.perf_counter() - n_start:.2f}s")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor("#d9d9d9")
    plt.gcf().set_facecolor("#d9d9d9")
    for n in n_values:
        avg = np.cumsum(mean_returns_by_n[n]) / (np.arange(len(mean_returns_by_n[n])) + 1)
        plt.plot(avg, label=f"n={n}")
    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.title("n-step SARSA (averaged over 1000 runs)")
    plt.ylim(-300, 0)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    _debug_print(debug, f"[experiment] saved plot to {out_path}")

    window = min(100, episodes)
    final100_by_n = {n: float(mean_returns_by_n[n][-window:].mean()) for n in n_values}
    best_n = max(final100_by_n, key=final100_by_n.get)

    print("Final-100 episode mean return by n:")
    for n in n_values:
        print(f"  n={n}: {final100_by_n[n]:.3f}")
    print(f"Best n (final-{window} mean): n={best_n}")
    print(f"Saved plot: {out_path}")
    if visualize_policy:
        q_visualizer(world, mean_q_by_n[1], title=f"Best policy for n={1}:")
        q_visualizer(world, mean_q_by_n[2], title=f"Best policy for n={2}:")
        q_visualizer(world, mean_q_by_n[3], title=f"Best policy for n={3}:")
        q_visualizer(world, mean_q_by_n[4], title=f"Best policy for n={4}:")

    _debug_print(debug, f"[experiment] completed in {time.perf_counter() - experiment_start:.2f}s")

    return {
        "mean_returns_by_n": mean_returns_by_n,
        "mean_q_by_n": mean_q_by_n,
        "final100_by_n": final100_by_n,
        "plot_path": out_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cliff World n-step SARSA experiment and visualize learned policy.")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per run (default: 500)")
    parser.add_argument("--runs", type=int, default=1000, help="Number of runs averaged per n (default: 1000)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate alpha (default: 0.5)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon-greedy exploration rate (default: 0.1)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor gamma (default: 1.0)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode (default: 1000)")
    parser.add_argument("--seed-base", type=int, default=123, help="Seed base for deterministic runs (default: 123)")
    parser.add_argument(
        "--out",
        type=str,
        default="sarsa/artifacts/cliff_nstep_returns.png",
        help="Output plot path (default: sarsa/artifacts/cliff_nstep_returns.png)",
    )
    parser.add_argument(
        "--no-visualizer",
        action="store_true",
        help="Disable terminal policy visualization.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug tracing for runs, episodes, and step heartbeats.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run_cliff_nstep_experiment(
        n_values=(1, 2, 3, 4),
        episodes=args.episodes,
        runs_per_n=args.runs,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        max_steps=args.max_steps,
        seed_base=args.seed_base,
        out_path=args.out,
        visualize_policy=not args.no_visualizer,
        debug=args.debug,
    )
    run_validation_checks(results, episodes=args.episodes)
