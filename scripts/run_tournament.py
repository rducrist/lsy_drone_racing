from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import fire
import toml
from run_pair_eval import run as run_pair_eval


def _load_teams(team_file: Path) -> list[dict]:
    if not team_file.exists():
        raise FileNotFoundError(f"Team file does not exist: {team_file}")

    data = toml.load(team_file)
    teams = data.get("team", [])
    if not isinstance(teams, list) or len(teams) < 2:
        raise ValueError("Team file must contain at least two [[team]] entries.")

    normalized = []
    for index, team in enumerate(teams):
        name = team.get("name")
        path = team.get("path")
        if not name or not path:
            raise ValueError(f"Team entry #{index} must define 'name' and 'path'.")

        normalized.append(
            {
                "name": str(name),
                "path": str(Path(path).resolve()),
                "controller": str(team.get("controller", "attitude_controller.py")),
            }
        )
    return normalized


def _pair_key(team_a: str, team_b: str) -> str:
    return f"{team_a}__vs__{team_b}".replace("/", "_")


def run(
    team_file: str,
    sim_repo: str,
    leaderboard_file: str = "leaderboard.toml",
    work_dir: str = "tournament_results",
    sim_config: str = "multi_level0.toml",
    n_runs: int = 1,
    action_timeout: float = 10.0,
    startup_delay: float = 3.0,
    timeout: float = 300.0,
    ros_domain_id_base: int = 0,
):
    """Run all unique team pairs from a TOML file and store results in leaderboard TOML."""
    team_file_path = Path(team_file)
    sim_repo_path = Path(sim_repo).resolve()
    leaderboard_path = Path(leaderboard_file)
    work_dir_path = Path(work_dir)
    pair_results_dir = work_dir_path / "pair_results"
    pair_logs_dir = work_dir_path / "logs"

    teams = _load_teams(team_file_path)

    work_dir_path.mkdir(parents=True, exist_ok=True)
    pair_results_dir.mkdir(parents=True, exist_ok=True)
    pair_logs_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)

    leaderboard = {
        "metadata": {
            "team_file": str(team_file_path.resolve()),
            "sim_repo": str(sim_repo_path),
            "n_teams": len(teams),
        }
    }

    for pair_index, (team_a, team_b) in enumerate(combinations(teams, 2)):
        key = _pair_key(team_a["name"], team_b["name"])
        pair_output = pair_results_dir / f"{key}.json"
        pair_logs = pair_logs_dir / key

        print(f"Running pair {pair_index + 1}: {team_a['name']} vs {team_b['name']}")
        run_pair_eval(
            sim_repo=str(sim_repo_path),
            controller_a_repo=team_a["path"],
            controller_b_repo=team_b["path"],
            output_file=str(pair_output),
            team_a=team_a["name"],
            team_b=team_b["name"],
            logs_dir=str(pair_logs),
            sim_config=sim_config,
            n_runs=n_runs,
            action_timeout=action_timeout,
            startup_delay=startup_delay,
            timeout=timeout,
            controller_a=team_a["controller"],
            controller_b=team_b["controller"],
            ros_domain_id=ros_domain_id_base + pair_index,
        )

        pair_result = json.loads(pair_output.read_text())
        leaderboard[key] = {
            "team_a": pair_result["team_a"],
            "team_b": pair_result["team_b"],
            "controller_a_repo": pair_result["controller_a_repo"],
            "controller_b_repo": pair_result["controller_b_repo"],
            "average_time_a": pair_result["average_time_a"],
            "average_time_b": pair_result["average_time_b"],
            "winner": pair_result["winner"],
            "episode_times": pair_result["episode_times"],
            "episode_finished": pair_result["episode_finished"],
            "logs_dir": pair_result["logs_dir"],
        }

        with open(leaderboard_path, "w") as file:
            toml.dump(leaderboard, file)

    print(f"Wrote leaderboard to {leaderboard_path.resolve()}")


if __name__ == "__main__":
    fire.Fire(run)
