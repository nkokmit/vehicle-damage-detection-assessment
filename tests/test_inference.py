from src.inference.postprocess import filter_predictions


def test_filter_predictions_by_score() -> None:
    predictions = [{"score": 0.2}, {"score": 0.9}, {}]
    result = filter_predictions(predictions, score_threshold=0.5)
    assert result == [{"score": 0.9}, {}]
