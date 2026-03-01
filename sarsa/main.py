import argparse

from n_step_sarsa import run_cliff_nstep_experiment, run_validation_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run n-step SARSA on Cliff World (n=1..4).")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per run (default: 500)")
    parser.add_argument("--runs", type=int, default=1000, help="Independent seeds per n (default: 1000)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate alpha (default: 0.5)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon-greedy exploration rate (default: 0.1)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor gamma (default: 1.0)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode (default: 1000)")
    parser.add_argument("--seed-base", type=int, default=123, help="Seed base for reproducibility (default: 123)")
    parser.add_argument(
        "--out",
        type=str,
        default="sarsa/artifacts/cliff_nstep_returns.png",
        help="Output PNG path (default: sarsa/artifacts/cliff_nstep_returns.png)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug tracing from n_step_sarsa.",
    )
    parser.add_argument(
        "--no-visualizer",
        action="store_true",
        help="Disable terminal policy visualization.",
    )
    parser.add_argument(
        "--policy-source",
        choices=("eval_greedy", "single_run", "mean_q"),
        default="eval_greedy",
        help="Policy visualization source (default: eval_greedy).",
    )
    parser.add_argument(
        "--tie-tol",
        type=float,
        default=1e-6,
        help="Tolerance for marking near-equal best actions as ties (default: 1e-6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
        policy_source=args.policy_source,
        tie_tol=args.tie_tol,
        debug=args.debug,
    )
    
    run_validation_checks(results, episodes=args.episodes)


if __name__ == "__main__":
    main()
