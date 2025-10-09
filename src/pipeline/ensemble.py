THRESHOLD_LOW = 500  # 30th percentile
THRESHOLD_MED = 3000  # 60th percentile
THRESHOLD_HIGH = 15000  # 80th percentile


def get_weights(num_scoring: int):

    if num_scoring < THRESHOLD_LOW:
        # Very few ratings - unreliable site mean
        # Trust global model more, personal taste matters
        wp, wg, ws = 0.45, 0.45, 0.10
    elif num_scoring < THRESHOLD_MED:
        # Few ratings - site mean becoming reliable
        # Balance between all three
        wp, wg, ws = 0.40, 0.35, 0.25
    elif num_scoring < THRESHOLD_HIGH:
        # Good amount of ratings - site mean reliable
        # Reduce personal bias
        wp, wg, ws = 0.30, 0.25, 0.45
    else:
        # Many ratings - site mean very reliable
        # Minimize personal bias, trust the crowd
        wp, wg, ws = 0.25, 0.15, 0.60

    return {"wp": wp, "wg": wg, "ws": ws}