"""
Script that will evaluate the performance of an agent on the specified environments. The agent is imported from
air_hockey_agent/agent_builder.py, where it should be implemented.

Args:
    -c --config path/to/config.yml:  Specify the path to the config.yml that should be used. The default config is in
            air_hockey_agent/agent_config.yml. The config specifies all parameters for the evaluation. For convenience
            the environment_list, number_of_experiments and log_dir can be overriden with the args below
    -e --env "tournament":    Overrides the env parameter of the config. Specifies the environments
            which are evaluated. Multiple environments can be provided. All possible envs are: ["hit", "defend", 
            "prepare", "tournament", "tournament_server"]
    -r --render: Set the flag to spawn a viewer which renders the simulation, Overrides the render param of the config
    -n --n_episodes 50: specify the number of episodes used for evaluation
    --n_cores 4: specify the amount of CPU cores used for evaluation. Set to -1 to use all available cores.
    --log_dir: Specify the path to the log directory, Overrides the log_dir of the config
    --example baseline: Load the provided example agents. Can either be "baseline" for the traditional robotics
            solution or "sac" for an end-to-end trained solution with SAC.

Examples:
    To view the baseline Agent on the hit environment:
    python run.py --example baseline -e hit -n 1 -r

    To generate a report for the first phase:
    python run.py -e tournament -g phase-1

    Of course all these parameters can also just be set in a config file which is loaded by via the
     --config /path/to/conf.yml. Or just modify the default config at air_hockey_agent/agent_config.yml

"""

from datetime import datetime
import os
from argparse import ArgumentParser
from pathlib import Path

import yaml

from air_hockey_challenge.framework.evaluate_agent import evaluate
from air_hockey_challenge.framework.evaluate_tournament import run_tournament
from air_hockey_challenge.utils.tournament_agent_server import run_tournament_server


def get_args():
    parser = ArgumentParser()
    arg_test = parser.add_argument_group("override parameters")

    env_choices = ["hit", "defend", "prepare", "tournament", "tournament_server"]

    arg_test.add_argument(
        "-e",
        "--env",
        nargs="+",
        choices=env_choices,
        help="Environments to be used.",
    )

    arg_test.add_argument(
        "--n_cores", type=int, help="Number of CPU cores used for evaluation."
    )

    arg_test.add_argument(
        "-n",
        "--n_episodes",
        type=int,
        help="Each seed will run for this number of Episodes.",
    )

    arg_test.add_argument(
        "--steps_per_game",
        type=int,
        help="Number of steps per game",
    )

    arg_test.add_argument(
        "--log_dir", type=str, help="The directory in which the logs are written"
    )

    arg_test.add_argument(
        "--example",
        type=str,
        choices=["hit-agent", "defend-agent", "baseline", "ppo_baseline", "atacom"],
        default="",
    )

    default_path = Path(__file__).parent.joinpath("air_hockey_agent/agent_config.yml")
    arg_test.add_argument(
        "-c",
        "--config",
        type=str,
        default=default_path,
        help="Path to the config file.",
    )

    arg_test.add_argument(
        "-r", "--render", action="store_true", help="If set renders the environment"
    )

    arg_test.add_argument(
        "--host", type=str, help="Host IP for tournament agent server"
    )
    arg_test.add_argument(
        "--port", type=int, help="Host port for tournament agent server"
    )

    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    args = get_args()

    # Remove all None entries
    filtered_args = {k: v for k, v in args.items() if v is not None}

    # Load config
    if os.path.exists(filtered_args["config"]):
        with open(filtered_args["config"]) as stream:
            config = yaml.safe_load(stream)
    else:
        print("Could not read config file with path: ", filtered_args["config"])
        config = {"quiet": False, "render": False}
    del filtered_args["config"]

    # Load Agent
    if filtered_args["example"] == "":
        from air_hockey_agent.agent_builder import build_agent
    elif filtered_args["example"] == "hit-agent":
        from examples.control.hitting_agent import build_agent
    elif filtered_args["example"] == "defend-agent":
        from examples.control.defending_agent import build_agent
    elif filtered_args["example"] == "baseline":
        from baseline.baseline_agent.baseline_agent import build_agent
    elif filtered_args["example"] == "ppo_baseline":
        from baseline.ppo_baseline_agent.agent import build_agent
    elif filtered_args["example"] == "atacom":
        from examples.rl.agent_loader import build_agent
    del filtered_args["example"]

    # Update config with command line args
    config.update(filtered_args)
    config["env_list"] = config["env"]
    del config["env"]

    config["log_dir"] = Path(config["log_dir"]) / datetime.now().strftime("eval-%Y-%m-%d_%H-%M-%S")

    if "tournament" in config["env_list"]:
        assert config["env_list"] == ["tournament"], "The tournament environment can only be used alone"
        run_tournament(build_agent, **config)
    elif "tournament_server" in config["env_list"]:
        assert config["env_list"] == ["tournament_server"], "The tournament server environment can only be used alone"
        run_tournament_server(build_agent, **config)
    else:
        assert config["env_list"], "At least one environment has to be specified"
        evaluate(build_agent, **config)
