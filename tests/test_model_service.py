from services.model_service import model_version


# --- NFR-4: reproducibility -- a stored prediction must be tied to a
# version hash of the feature set/hyperparameters that produced it, same
# config_version()/weights_version() pattern used elsewhere this session.

def test_model_version_is_stable_for_the_same_inputs():
    assert model_version(tune=False) == model_version(tune=False)


def test_model_version_differs_between_tune_and_no_tune():
    # tune is part of what the stored row needs to regenerate the exact
    # training run (GridSearchCV vs a single fit) -- must change the hash.
    assert model_version(tune=False) != model_version(tune=True)
