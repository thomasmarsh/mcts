from ConfigSpace import ConfigurationSpace, EqualsCondition
import smac
from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Callback, Scenario
from pathlib import Path
from ConfigSpace import Categorical, Float, Integer, Constant
from smac.runhistory import TrialInfo, TrialValue
from smac.utils.configspace import get_config_hash

import math
import os

def find_root():
    return Path(__file__).parent.parent

class CustomCallback(Callback):
    def __init__(self) -> None:
        self.last = None
        return None

    def on_end(self, smbo: smac.main.smbo.SMBO):
        return None
    

    def on_next_configurations_end(self, config_selector, config):
        print(config)
        return None

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        incumbent = smbo.intensifier.get_incumbent()
        smbo.intensifier._n_seeds
        assert incumbent is not None
        hash = get_config_hash(incumbent)
        if hash != self.last:
            cost = smbo.runhistory.get_cost(incumbent)
            print(f"hash={hash} cost={cost} dict={dict(incumbent)}")
            print("")
            self.last = hash
        return None

class GameSearch:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        bias = Float('bias', (0, 10), default=10e-6)
        c = Float('c', (0, 3), default=math.sqrt(2))
        epsilon = Float('epsilon', (0, 1), default=0.1)
        final_action = Constant('final-action', 'robust_child')
        k = Integer('k', (0, 2000))
        q_init = Categorical('q-init', ['Draw', 'Infinity', 'Loss', 'Parent', 'Win'])
        rave = Integer('rave', (0, 2000))
        schedule = Categorical('schedule', ['hand_selected', 'min_mse', 'threshold'])
        threshold = Integer('threshold', (0, 2000), default=700)

        cs.add_hyperparameters([bias, c, epsilon, final_action, k, q_init, rave, schedule, threshold])

        is_hand_selected = EqualsCondition(k, schedule, 'hand_selected')
        is_min_mse = EqualsCondition(bias, schedule, 'min_mse')
        is_threshold = EqualsCondition(rave, schedule, 'threshold')

        cs.add_condition(is_hand_selected)
        cs.add_condition(is_min_mse)
        cs.add_condition(is_threshold)

        # final_action = Categorical('final-action', ['max_avg', 'robust_child', 'secure_child'])
        # a = Float('a', (0,40), default=4.0)
        # cond = EqualsCondition(a, final_action, 'secure_child')
        # cs.add_condition(cond)
        print(dict(cs))

        return cs

    def train(self) -> str:
        return str(find_root() / 'target' / 'release' / 'hyper')


if __name__ == "__main__":
    model = GameSearch()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(
        model.configspace,
        deterministic=False,
        n_trials=1000,
        n_workers=(os.cpu_count() // 2))

    Facade = HyperparameterOptimizationFacade
    
    config_selector = Facade.get_config_selector(scenario, retrain_after=1)

    # Now we use SMAC to find the best hyperparameters
    smac = Facade(
        scenario,
        model.train(),
        logging_level=20,
        config_selector=config_selector,
        callbacks=[CustomCallback()],
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
    )

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"default cost={default_cost}")
