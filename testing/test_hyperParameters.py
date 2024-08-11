import pytest
import optuna
import GTS


def testSpace():
    ts = GTS.TrainingSpace(GTS.PhysicalModelID.invertedPendulum, "test",
                           "reinforce", 100, 20, {"y": 1},
                           {"x": {
                               "min": -10.0,
                               "max": 10.0
                           }})

    def objective(trial: optuna.trial.Trial) -> float:
        spec = ts.bake(trial)
        x = spec.params["x"]
        assert isinstance(x, float)
        return -(x - 2) * x

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    found_x = best_params["x"]
    #print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))
    assert 1.0 == pytest.approx(found_x, 0.05)
