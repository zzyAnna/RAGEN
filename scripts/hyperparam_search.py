from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Optional
import itertools
import subprocess
import os
from datetime import datetime
import argparse
from enum import Enum, auto
import re
import json
from pathlib import Path


# ============= Hyperparameter Configuration Part =============
class ParamType(Enum):
    NUMERIC = auto()
    CATEGORICAL = auto()
    BOOLEAN = auto()

@dataclass
class HyperParam:
    name: str
    param_type: ParamType
    default_value: Any
    search_space: Optional[List[Any]] = None
    group: int = 0
    
    def get_values(self, mode: str) -> List[Any]:
        """Get parameter values based on mode (search/fixed/default)"""
        if mode == "search" and self.search_space:
            return self.search_space
        return [self.default_value]

@dataclass
class HyperParamGroup:
    group_id: int
    params: List[HyperParam]
    description: str = ""

class HyperParamConfig:
    def __init__(self):
        self.groups: Dict[int, HyperParamGroup] = {}
        self.fixed_params: Dict[str, Any] = {
            "model.base_model": ["Qwen/Qwen2.5-0.5B-Instruct"],
            "optimization.adv_estimator": ["grpo"],
            "training.use_kl_loss": [True]
        }
        self._initialize_groups()

    def _initialize_groups(self):
        """Initialize all hyperparameter groups"""
        self.groups = {
            1: HyperParamGroup(1, [
                HyperParam("training.ppo_batch_size", ParamType.NUMERIC, 64, 
                          search_space=[16, 32, 64, 128, 256], group=1)
            ], "PPO Batch Size"),
            
            2: HyperParamGroup(2, [
                HyperParam("training.train_batch_size", ParamType.NUMERIC, 8,
                          search_space=[8, 32, 64, 128, 256], group=2),
                HyperParam("training.n_rollout", ParamType.NUMERIC, 16,
                          search_space=[1, 2, 4, 8, 16], group=2)
            ], "Training Batch Size and Rollout"),
            
            3: HyperParamGroup(3, [
                HyperParam("training.kl_coef", ParamType.NUMERIC, 0.04,
                          search_space=[0.001, 0.005, 0.01, 0.04, 0.1, 0.5], group=3)
            ], "KL Coefficient"),
            
            4: HyperParamGroup(4, [
                HyperParam("training.max_turns", ParamType.NUMERIC, 5,
                          search_space=[2, 5, 8], group=4),
                HyperParam("training.temperature", ParamType.NUMERIC, 1.0,
                          search_space=[0, 0.5, 1], group=4)
            ], "Max Turns and Temperature"),
            
            5: HyperParamGroup(5, [
                HyperParam("training.actor_lr", ParamType.NUMERIC, 1e-6,
                          search_space=[1e-6, 5e-6, 1e-5], group=5)
            ], "Actor Learning Rate")
        }


# ============= Hyperparameter Search Part =============
class HyperParamSearch:
    def __init__(self, config: HyperParamConfig):
        self.config = config
        self.param_grid: Dict[str, List[Any]] = {}
        self.searching_params: List[str] = []
        self.log_name = ""

    def setup_search_group(self, search_group: int, fixed_values: Dict[str, Any]) -> None:
        """Set up parameter grid based on search group and fixed values"""
        # Add fixed parameters
        self.param_grid.update(self.config.fixed_params)
        
        # Process each group based on its relation to the search group
        for group_id, group in self.config.groups.items():
            if group_id < search_group:
                # Fixed values for previous groups
                for param in group.params:
                    if param.name in fixed_values:
                        self.param_grid[param.name] = [fixed_values[param.name]]
                    else:
                        raise ValueError(f"Missing fixed value for {param.name} in group {group_id}")
            
            elif group_id == search_group:
                # Search space for current group
                for param in group.params:
                    if param.search_space:
                        self.param_grid[param.name] = param.search_space
                        self.searching_params.append(param.name)
            
            else:
                # Default values for future groups
                for param in group.params:
                    self.param_grid[param.name] = [param.default_value]

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameter combinations"""
        if all(key in params for key in ["training.ppo_batch_size", "training.train_batch_size", "training.n_rollout"]):
            if params["training.ppo_batch_size"] > params["training.train_batch_size"] * params["training.n_rollout"]:
                print(f"Invalid combination: ppo_batch_size ({params['training.ppo_batch_size']}) > "
                      f"train_batch_size ({params['training.train_batch_size']}) * "
                      f"n_rollout ({params['training.n_rollout']})")
                return False
        return True

    def generate_command(self, params: Dict[str, Any], base_name: str, env_name: str = "sokoban") -> str:
        """Generate command string for running experiment"""
        param_parts = []
        for param_name in self.searching_params:
            if param_name in params:
                value = params[param_name]
                param_name_short = param_name.split('.')[-1]
                formatted_value = self._format_value(value)
                param_parts.append(f"{param_name_short}_{formatted_value}")

        timestamp = datetime.now().strftime("%Hh%Mm%Ssec")
        self.log_name = f"[hs]_{'_'.join(param_parts)}"
        experiment_name = f"{base_name}_{'_'.join(param_parts)}_{timestamp}"

        cmd_parts = [
            f"bash train.sh {env_name}",
            f"model.experiment_name={experiment_name}"
        ]
        
        for key, value in params.items():
            cmd_parts.append(f"{key}={value}")

        return " \\\n".join(cmd_parts)

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format parameter value for command string"""
        if isinstance(value, float):
            if value < 0.0001:
                return f"{value:.2e}".replace('-0', '-').replace('+0', '')
            return f"{value:.4f}".rstrip('0').rstrip('.')
        return str(value)

    def run_grid_search(self, base_experiment_name: str, dry_run: bool = True, env_name: str = "sokoban") -> None:
        """Run grid search with all parameter combinations"""
        combinations = [dict(zip(self.param_grid.keys(), vals)) 
                       for vals in itertools.product(*self.param_grid.values())]
        
        print(f"Total combinations: {len(combinations)}")
        
        log_dir = self._setup_log_directory()
        
        for i, params in enumerate(combinations, 1):
            if not self.validate_params(params):
                print(f"Skipping invalid combination: {params}")
                print('-' * 80)
                continue
                
            command = self.generate_command(params, base_experiment_name, env_name)
            print(f"Running combination {i}/{len(combinations)}:")
            print(command)
            print('-' * 80)
            
            if not dry_run:
                self._run_experiment(command, log_dir)

    def _setup_log_directory(self) -> str:
        """Set up logging directory"""
        log_dir = "./log/terminal/hyper_param_search_logs/"
        
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _run_experiment(self, command: str, log_dir: str) -> None:
        """Run single experiment with logging"""
        try:
            log_file = os.path.join(log_dir, 
                                  f"{self.log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            with open(log_file, 'w') as f:
                subprocess.run(command, shell=True, check=True, stdout=f, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}, return code was {e.returncode}")
            if hasattr(e, 'output'):
                print(e.output)



# ========== Hyperparameter Results Tracking Part =============

@dataclass
class ExperimentResult:
    params: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str
    experiment_name: str
    
    @property
    def success_rate(self) -> float:
        """Get the primary metric (success rate) for this experiment"""
        return self.metrics.get('success_rate', 0.0)

class ResultsTracker:
    def __init__(self, base_dir: str = "./log/terminal/hyper_param_search_logs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.base_dir / "search_results.json"
        self.best_params_file = self.base_dir / "best_params.json"
        self._initialize_files()

    def _initialize_files(self):
        """Initialize results and best params files if they don't exist"""
        if not self.results_file.exists():
            self._save_json(self.results_file, {"experiments": []})
        if not self.best_params_file.exists():
            self._save_json(self.best_params_file, {"best_params": {}})

    @staticmethod
    def _save_json(file_path: Path, data: Dict):
        """Save data to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _load_json(file_path: Path) -> Dict:
        """Load data from JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)

    def parse_log_file(self, log_file: Path) -> Optional[ExperimentResult]:
        """Parse a log file to extract experiment results and metrics"""
        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Extract experiment name
            exp_name_match = re.search(r'model\.experiment_name=([^\s]+)', content)
            if not exp_name_match:
                return None
            experiment_name = exp_name_match.group(1)

            # Extract parameters
            params = {}
            param_pattern = r'training\.([^\s=]+)=([^\s]+)'
            for match in re.finditer(param_pattern, content):
                param_name, value = match.groups()
                try:
                    # Convert string values to appropriate types
                    value = eval(value)  # safely convert numbers and booleans
                except:
                    pass  # keep as string if conversion fails
                params[f"training.{param_name}"] = value

            # Extract metrics (example: success rate)
            metrics = {}
            success_rate_match = re.search(r'Final Success Rate: (\d+\.?\d*)', content)
            if success_rate_match:
                metrics['success_rate'] = float(success_rate_match.group(1))
            else:
                return None  # Skip if no success rate found

            return ExperimentResult(
                params=params,
                metrics=metrics,
                timestamp=datetime.now().isoformat(),
                experiment_name=experiment_name
            )
        except Exception as e:
            print(f"Error parsing log file {log_file}: {e}")
            return None

    def process_new_logs(self, logs_dir: Path) -> List[ExperimentResult]:
        """Process all new log files in the specified directory"""
        results = []
        processed_logs = set(self._load_json(self.results_file).get("processed_logs", []))
        
        for log_file in logs_dir.glob("*.log"):
            if str(log_file) in processed_logs:
                continue
                
            result = self.parse_log_file(log_file)
            if result:
                results.append(result)
                processed_logs.add(str(log_file))
        
        # Update results file with new experiments
        current_data = self._load_json(self.results_file)
        current_data["experiments"].extend([
            {
                "params": r.params,
                "metrics": r.metrics,
                "timestamp": r.timestamp,
                "experiment_name": r.experiment_name
            }
            for r in results
        ])
        current_data["processed_logs"] = list(processed_logs)
        self._save_json(self.results_file, current_data)
        
        return results

    def update_best_parameters(self, group_id: int, results: List[ExperimentResult]):
        """Update best parameters for a specific group based on new results"""
        best_params = self._load_json(self.best_params_file)
        
        # Filter results for current group parameters
        group_params = self.get_group_parameters(group_id)
        group_results = [
            r for r in results
            if all(param in r.params for param in group_params)
        ]
        
        if not group_results:
            return
        
        # Find best result based on success rate
        best_result = max(group_results, key=lambda x: x.success_rate)
        
        # Update best parameters for this group
        best_params["best_params"][str(group_id)] = {
            param: best_result.params[param]
            for param in group_params
        }
        best_params["best_params"][str(group_id)]["success_rate"] = best_result.success_rate
        
        self._save_json(self.best_params_file, best_params)

    @staticmethod
    def get_group_parameters(group_id: int) -> List[str]:
        """Get parameter names for a specific group"""
        group_params = {
            1: ["training.ppo_batch_size"],
            2: ["training.train_batch_size", "training.n_rollout"],
            3: ["training.kl_coef"],
            4: ["training.max_turns", "training.temperature"],
            5: ["training.actor_lr"]
        }
        return group_params.get(group_id, [])

    def get_best_params_for_groups(self, groups: List[int]) -> Dict[str, Any]:
        """Get best parameters for specified groups"""
        best_params = self._load_json(self.best_params_file)
        result = {}
        
        for group in groups:
            group_best = best_params["best_params"].get(str(group))
            if group_best:
                # Exclude success_rate from parameters
                params = {k: v for k, v in group_best.items() if k != "success_rate"}
                result.update(params)
            else:
                raise ValueError(f"No best parameters found for group {group}")
                
        return result

class SearchOrchestrator:
    def __init__(self, results_tracker: ResultsTracker):
        self.tracker = results_tracker

    def run_search_round(self, search_group: int, env_name: str, exp_base_name: str, dry_run: bool = True):
        """Run a complete search round for a group"""
        # Get best parameters from previous groups
        previous_groups = list(range(1, search_group))
        try:
            fixed_params = self.tracker.get_best_params_for_groups(previous_groups)
        except ValueError as e:
            print(f"Error: {e}")
            print("Please complete previous group searches first.")
            return

        # Create and run search
        config = HyperParamConfig()  # From previous implementation
        search = HyperParamSearch(config)
        search.setup_search_group(search_group, fixed_params)
        
        # Run grid search
        search.run_grid_search(
            base_experiment_name=exp_base_name,
            dry_run=dry_run,
            env_name=env_name
        )
        
        if not dry_run:
            # Process results after experiments complete
            logs_dir = Path("./log/terminal/hyper_param_search_logs/")
            new_results = self.tracker.process_new_logs(logs_dir)
            if new_results:
                self.tracker.update_best_parameters(search_group, new_results)
            else:
                print("No new results found to process.")


# ============= Main Functionality Part =============
def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description='Hyperparameter search with group-based control')
    
    # Required arguments
    parser.add_argument('--env_name', type=str, required=True,
                      help='Environment name (e.g., sokoban)')
    parser.add_argument('--base_experiment_name', type=str, required=True,
                      help='Base name for the experiment')
    parser.add_argument('--search_group', type=int, required=True, choices=range(1, 6),
                      help='Group number to search (1-5)')
    
    # Optional arguments
    parser.add_argument('--dry_run', action='store_true',
                      help='Print commands without executing')
    
    # Group-specific parameters
    parser.add_argument('--ppo_batch_size', type=int,
                      help='Fixed value for PPO batch size (required for groups 2-5)')
    parser.add_argument('--train_batch_size', type=int,
                      help='Fixed value for training batch size (required for groups 3-5)')
    parser.add_argument('--n_rollout', type=int,
                      help='Fixed value for number of rollouts (required for groups 3-5)')
    parser.add_argument('--kl_coef', type=float,
                      help='Fixed value for KL coefficient (required for groups 4-5)')
    parser.add_argument('--max_turns', type=int,
                      help='Fixed value for max turns (required for group 5)')
    parser.add_argument('--temperature', type=float,
                      help='Fixed value for temperature (required for group 5)')
    
    return parser

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments"""
    if args.search_group > 1:
        if args.ppo_batch_size is None:
            raise ValueError("--ppo_batch_size required for groups 2-5")
    if args.search_group > 2:
        if args.train_batch_size is None or args.n_rollout is None:
            raise ValueError("--train_batch_size and --n_rollout required for groups 3-5")
    if args.search_group > 3:
        if args.kl_coef is None:
            raise ValueError("--kl_coef required for groups 4-5")
    if args.search_group > 4:
        if args.max_turns is None or args.temperature is None:
            raise ValueError("--max_turns and --temperature required for group 5")

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    validate_args(args)
    
    # Create config and search objects
    config = HyperParamConfig()
    search = HyperParamSearch(config)
    
    # Collect fixed values from arguments
    fixed_values = {}
    if args.ppo_batch_size is not None:
        fixed_values["training.ppo_batch_size"] = args.ppo_batch_size
    if args.train_batch_size is not None:
        fixed_values["training.train_batch_size"] = args.train_batch_size
    if args.n_rollout is not None:
        fixed_values["training.n_rollout"] = args.n_rollout
    if args.kl_coef is not None:
        fixed_values["training.kl_coef"] = args.kl_coef
    if args.max_turns is not None:
        fixed_values["training.max_turns"] = args.max_turns
    if args.temperature is not None:
        fixed_values["training.temperature"] = args.temperature
    
    # Set up search configuration
    search.setup_search_group(args.search_group, fixed_values)
    
    # Run the search
    search.run_grid_search(
        base_experiment_name=args.base_experiment_name,
        dry_run=args.dry_run,
        env_name=args.env_name
    )

if __name__ == "__main__":
    main()