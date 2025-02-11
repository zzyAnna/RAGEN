import itertools
from typing import List, Dict, Union, Tuple, Any
import subprocess
import os
from datetime import datetime
import argparse

searching_params = []
global_log_name = ""

class HyperParamSearch:
    def __init__(self) -> None:
        self.param_grid = {}
        self.log_name = ""
    
    def add_numeric_param(self, name: str, values: List[float]) -> None:
        self.param_grid[name] = values
    
    def add_categorical_param(self, name: str, values: List[str]) -> None:
        self.param_grid[name] = values

    def set_boolean_param(self, name: str, value: bool) -> None:
        self.param_grid[name] = [value]

    def add_boolean_param(self, name: str) -> None:
        self.param_grid[name] = [True, False]

    def generate_combinations(self) -> List[Dict[str, Union[float, str, bool]]]:
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combination)) for combination in combinations]

    def param_hard_checking(self, params) -> bool:
        '''Make sure ppo_batch_size is equal or smaller than train_batch_size * n_rollout'''
        assert params.get("training.ppo_batch_size") is not None and params.get("training.train_batch_size") is not None and params.get("training.n_rollout") is not None
        ppo_batch_size = params["training.ppo_batch_size"]
        train_batch_size = params["training.train_batch_size"]
        n_rollout = params["training.n_rollout"]
        if ppo_batch_size > train_batch_size * n_rollout:
            print(f"ppo_batch_size {ppo_batch_size} is larger than train_batch_size {train_batch_size} * n_rollout {n_rollout}")
            return False
        return True
    
    def generate_command(self, params: Dict[str, Union[float, str, bool]], base_name: str, env_name:str="sokoban") -> str:
        param_name_values = []
        my_params = {k: v for k, v in params.items() if k in searching_params}
        for key, value in my_params.items():
            # Extract the last part of the parameter name after the dot
            param_name = key.split('.')[-1]
            # Format the value based on its type
            if isinstance(value, float):
                # Handle scientific notation and round small numbers
                if value < 0.0001:
                    formatted_value = f"{value:.2e}".replace('-0', '-').replace('+0', '')
                else:
                    formatted_value = f"{value:.4f}".rstrip('0').rstrip('.')
            else:
                formatted_value = str(value)
            
            # Clean up the formatted value for use in filename
            # formatted_value = formatted_value.replace('.', 'p')  # Replace decimal points with 'p'
            param_name_values.append(f"{param_name}_{formatted_value}")
        
        # Join all parameter names and values
        params_str = '_'.join(param_name_values)
        timestamp = datetime.now().strftime("%Hh%Mm%Ssec")
        # Construct the full experiment name
        self.log_name = f"[hs]_{params_str}"
        experiment_name = f"{base_name}_{params_str}_{timestamp}"
        cmd_parts = [
            f"bash train.sh {env_name}",
            f"model.experiment_name={experiment_name}"
        ]
        for key, value in params.items():
            if isinstance(value, bool):
                cmd_parts.append(f"{key}={str(value)}")
            else:
                cmd_parts.append(f"{key}={value}")
                
        return " \\\n".join(cmd_parts)
    
    def run_grid_search(self, base_experiment_name: str, dry_run: bool=True, env_name: str="sokoban") -> None:
        combinations = self.generate_combinations()
        print(f"Total combinations: {len(combinations)}")

        # Check if we're in the RAGEN directory
        current_dir = os.getcwd()
        if not current_dir.endswith('RAGEN'):
            # Try to change to RAGEN directory
            try:
                if os.path.exists('RAGEN'):
                    log_dir = "./RAGEN/hyper_param_search_logs"
                elif os.path.exists('../RAGEN'):
                    log_dir = "../hyper_param_search_logs"
                else:
                    raise AssertionError("Must run from RAGEN directory or its parent directory")
            except Exception as e:
                raise AssertionError(f"Failed to find RAGEN directory: {e}")
        else:
            log_dir = "./hyper_param_search_logs"
        os.makedirs(log_dir, exist_ok=True)

        for i, params in enumerate(combinations, 1):
            command = self.generate_command(params, base_experiment_name, env_name)
            print(f"Running combination {i}/{len(combinations)}: {command}")
            # print(command)
            if not self.param_hard_checking(params):
                print("Skipping this combination due to hard checking failure.")
                print('-' * 80)
                continue
            print('-' * 80)
            if not dry_run:
                try:
                    log_file = os.path.join(log_dir, f"{self.log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                    with open(log_file, 'w') as f:
                        subprocess.run(command, shell=True, check=True, stdout=f, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred: {e}, return code was {e.returncode}")
                    print(e.output)
                    continue
        
        print(f"Total combinations: {len(combinations)}")

def assert_other_group_value_set(search: HyperParamSearch, search_group_id: int):
    all_values_dict = {
        1: ["training.ppo_batch_size"],
        2: ["training.train_batch_size", "training.n_rollout"],
        3: ["training.kl_coef"],
        4: ["training.max_turns", "training.temperature"],
        5: ["training.actor_lr"]
    }

    remaining_groups = [i for i in range(1, 6) if i != search_group_id]
    remaining_values = []
    for i in remaining_groups:
        remaining_values.extend(all_values_dict[i])
    assert all([search.param_grid.get(i) is not None for i in remaining_values]), f"Please set the values for {remaining_values} first"

# below we search different batch size
def search_group_1(search: HyperParamSearch, set_values: list=None) -> None: # type: ignore
    assert set_values is None or len(set_values) == 1
    if set_values is None:
        assert_other_group_value_set(search, 1)
        search.add_numeric_param("training.ppo_batch_size", [16, 32, 64, 128, 256])
        searching_params.append("training.ppo_batch_size")
        print("We are searching different ppo batch size...")
        print("...")
    else:
        search.add_numeric_param("training.ppo_batch_size", [set_values[0]])

# below we search different batch size. We all know that n_rollout*training_batch_size means the data sampled from the rollout
## train batch size means the prompt trajectory
def search_group_2(search: HyperParamSearch, set_values: list=None) -> None: # type: ignore
    assert set_values is None or len(set_values) == 2
    if set_values is None:
        assert_other_group_value_set(search, 2)
        search.add_numeric_param("training.train_batch_size", [8, 32, 64, 128, 256])
        search.add_numeric_param("training.n_rollout", [1, 2, 4, 8, 16])
        searching_params.append("training.train_batch_size")
        searching_params.append("training.n_rollout")
        print("We are searching different train batch size and n_rollout ...")
        print("...")
    else:
        search.add_numeric_param("training.train_batch_size", [set_values[0]])
        search.add_numeric_param("training.n_rollout", [set_values[1]])


# below we search different kl coef
def search_group_3(search: HyperParamSearch, set_values: list=None) -> None: # type: ignore
    assert set_values is None or len(set_values) == 1
    if set_values is None:
        assert_other_group_value_set(search, 3)
        search.add_numeric_param("training.kl_coef", [0.001, 0.005, 0.01, 0.04, 0.1, 0.5])
        searching_params.append("training.kl_coef")
        print("We are searching different kl coef...")
        print("...")
    else:
        search.add_numeric_param("training.kl_coef", [set_values[0]])


# below we search different max turns and temperature
def search_group_4(search: HyperParamSearch, set_values: list=None) -> None: # type: ignore
    assert set_values is None or len(set_values) == 2
    if set_values is None:
        assert_other_group_value_set(search, 4)
        search.add_numeric_param("training.max_turns", [2, 5, 8])
        search.add_numeric_param("training.temperature", [0, 0.5, 1])
        searching_params.append("training.max_turns")
        searching_params.append("training.temperature")
        print("We are searching different max turns and temperature ...")
        print("...")
    else:
        search.add_numeric_param("training.max_turns", [set_values[0]])
        search.add_numeric_param("training.temperature", [set_values[1]])

# below we search different actor learning rate
def search_group_5(search: HyperParamSearch, set_values: list=None) -> None: # type: ignore
    assert set_values is None or len(set_values) == 1
    if set_values is None:
        assert_other_group_value_set(search, 5)
        search.add_numeric_param("training.actor_lr", [1e-6, 5e-6, 1e-5])
        searching_params.append("training.actor_lr")
        print("We are searching different actor learning rate...")
        print("...")
    else:
        search.add_numeric_param("training.actor_lr", [set_values[0]])

def set_default_value_by_groups(search: HyperParamSearch, wanted_groups: list[int]):
    '''This function will set the default value for wanted groups'''
    assert all([i in range(1, 6) for i in wanted_groups])
    default_dict = {
        1: [("training.ppo_batch_size", 64)],
        2: [("training.train_batch_size", 8), ("training.n_rollout", 16)],
        3: [("training.kl_coef", 0.04)],
        4: [("training.max_turns", 5), ("training.temperature", 1.0)],
        5: [("training.actor_lr", 1e-6)]
    }

    # Process each wanted group
    for group_num in wanted_groups:
        group_defaults = default_dict[group_num]
        for param_name, value in group_defaults:
            search.add_numeric_param(param_name, [value])

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter search script with group-specific controls')
    
    # Basic arguments
    parser.add_argument('--base_experiment_name', type=str, default='hyper_param_search', required=True,
                        help='Base name for the experiment')
    parser.add_argument('--env_name', type=str, default='sokoban', required=True,
                        help='Environment name (default: sokoban)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only print commands without executing them')
    
    # Search group selection
    parser.add_argument('--search_group', type=int, choices=[1, 2, 3, 4, 5], default=1, required=True,
                        help='The group of hyperparameters to search:\n'
                             '1: ppo_batch_size\n'
                             '2: train_batch_size, n_rollout\n'
                             '3: kl_coef\n'
                             '4: max_turns, temperature\n'
                             '5: actor_lr')
    
    # Group 1 settings
    parser.add_argument('--ppo_batch_size', type=int,
                        help='Set fixed value for ppo_batch_size (Group 1)')
    
    # Group 2 settings
    parser.add_argument('--train_batch_size', type=int,
                        help='Set fixed value for train_batch_size (Group 2)')
    parser.add_argument('--n_rollout', type=int,
                        help='Set fixed value for n_rollout (Group 2)')
    
    # Group 3 settings
    parser.add_argument('--kl_coef', type=float,
                        help='Set fixed value for kl_coef (Group 3)')
    
    # Group 4 settings
    parser.add_argument('--max_turns', type=int,
                        help='Set fixed value for max_turns (Group 4)')
    parser.add_argument('--temperature', type=float,
                        help='Set fixed value for temperature (Group 4)')
    
    # Group 5 settings
    parser.add_argument('--actor_lr', type=float,
                        help='Set fixed value for actor_lr (Group 5)')
    
    args = parser.parse_args()
    
    # Validate arguments based on search group
    if args.search_group > 1:
        # Check if all parameters from previous groups are set
        if args.search_group > 1 and args.ppo_batch_size is None:
            parser.error("When searching group > 1, --ppo_batch_size must be set")
        if args.search_group > 2 and (args.train_batch_size is None or args.n_rollout is None):
            parser.error("When searching group > 2, --train_batch_size and --n_rollout must be set")
        if args.search_group > 3 and args.kl_coef is None:
            parser.error("When searching group > 3, --kl_coef must be set")
        if args.search_group > 4 and (args.max_turns is None or args.temperature is None):
            parser.error("When searching group > 4, --max_turns and --temperature must be set")
    
    return args

def main():
    args = parse_args()
    search = HyperParamSearch()
    
    # Add fixed parameters
    search.add_categorical_param("model.base_model", ["Qwen/Qwen2.5-0.5B-Instruct"])
    search.add_categorical_param("optimization.adv_estimator", ["grpo"])
    search.set_boolean_param("training.use_kl_loss", True)
    
    # Set parameters based on search group
    search_funcs = {
        1: lambda: search_group_1(search),
        2: lambda: search_group_2(search),
        3: lambda: search_group_3(search),
        4: lambda: search_group_4(search),
        5: lambda: search_group_5(search)
    }
    
    # Set values for groups before search group
    if args.search_group > 1:
        search_group_1(search, [args.ppo_batch_size])
    if args.search_group > 2:
        search_group_2(search, [args.train_batch_size, args.n_rollout])
    if args.search_group > 3:
        search_group_3(search, [args.kl_coef])
    if args.search_group > 4:
        search_group_4(search, [args.max_turns, args.temperature])
    
    # Set default values for groups after search group
    default_groups = list(range(args.search_group + 1, 6))
    if default_groups:
        set_default_value_by_groups(search, default_groups)

    # Run search for current group
    search_funcs[args.search_group]()
    
    
    
    # Run the grid search
    search.run_grid_search(
        base_experiment_name=args.base_experiment_name,
        dry_run=args.dry_run,
        env_name=args.env_name
    )

if __name__ == "__main__":
    main()