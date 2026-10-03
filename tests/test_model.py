from src.models.model_factory import create_model


def test_create_model_default() -> None:
    model = create_model()
    assert model.config.num_classes == 2
