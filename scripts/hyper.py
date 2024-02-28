from ConfigSpace import ConfigurationSpace
from smac import BlackBoxFacade, Scenario
from pathlib import Path
from ConfigSpace import Categorical, Float, Integer

import math
import os


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

def find_root():
    return Path(__file__).parent.parent

class GameSearch:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        c = Float('c', (0, 3), default=math.sqrt(2))
        #bias = Float('bias', (0, 1000), default=10e-6)
        #threshold = Integer('threshold', (0, 1000), default=100)
        #epsilon = Float('epsilon', (0, 1), default=0.1)
        q_init = Categorical("q-init", ["Draw", "Infinity", "Loss", "Parent", "Win"])
        #cs.add_hyperparameters([c, bias, threshold, epsilon, q_init])
        cs.add_hyperparameters([c, q_init])
        return cs

    def train(self) -> str:
        return str(find_root() / "target" / "release" / "hyper")


if __name__ == "__main__":
    model = GameSearch()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(
        model.configspace,
        deterministic=True,
        n_trials=100,
        n_workers=(os.cpu_count() // 2))
    
    # Now we use SMAC to find the best hyperparameters
    smac = BlackBoxFacade(
        scenario,
        model.train(),
        logging_level=20,
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
    )
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
    print(incumbent)
