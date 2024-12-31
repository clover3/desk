import logging
from typing import Dict, Any
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterSampler

from toxicity.io_helper import init_logging
from toxicity.open_ai_mod.train_proto import protonet_train_exp
from toxicity.reddit.proto.protory_net2 import ProtoryNet2

LOG = logging.getLogger(__name__)


def run_random_search(conf_path: str, n_iter: int = 10) -> Dict[str, Any]:
    """
    Run random hyperparameter search and return best configuration.

    Args:
        conf_path: Path to base configuration file
        n_iter: Number of random combinations to try

    Returns:
        Dictionary containing best parameters and corresponding performance
    """
    # Define parameter space
    param_distributions = {
        'lr': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'epoch': [1, 2, 4, 8, 16, 32, 64, 128],
        "k_protos": [5, 10, 20, 40, 80],
        "alpha": [0, 0.0001, 0.01],
        "beta": [0, 0.01, 0.1]
    }

    # Sample random parameter combinations
    param_combinations = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))

    # Track best performance
    best_performance = float('-inf')
    best_params = None
    best_conf = None

    # Load base configuration
    base_conf = OmegaConf.load(conf_path)

    # Try each parameter combination
    for params in param_combinations:
        # Create a new config for each run to avoid modifying the base config
        current_conf = OmegaConf.create(base_conf)

        # Update configuration with current parameters
        run_name = "_".join([f"{k[:1]}{v}" for k, v in params.items()])
        current_conf["run_name"] = run_name

        # Update all parameters in the config
        for k, v in params.items():
            current_conf[k] = v

        LOG.info(f"Testing parameters: {params}")
        LOG.info(str(current_conf))

        try:
            # Run training with current parameters
            performance = protonet_train_exp(ProtoryNet2, current_conf, False)

            # Update best if current performance is better
            if performance > best_performance:
                best_performance = performance
                best_params = params
                best_conf = current_conf.copy()

            LOG.info(f"Performance: {performance}")
            LOG.info(f"Best so far: {best_performance}")

        except Exception as e:
            LOG.error(f"Error during training with parameters {params}: {str(e)}")
            continue

    return {
        'best_params': best_params,
        'best_performance': best_performance,
        'best_conf': best_conf
    }


if __name__ == "__main__":
    init_logging()

    conf_path = "confs/proto/open_ai_mod1.yaml"

    # Run random search with 10 iterations
    results = run_random_search(conf_path, n_iter=10)

    LOG.info("Random Search Complete")
    LOG.info(f"Best parameters found: {results['best_params']}")
    LOG.info(f"Best performance: {results['best_performance']}")
    LOG.info(f"Best configuration: {results['best_conf']}")