import enum


class TrainPhase(enum.Enum):
    HEATMAP = "HEATMAP"
    REGRESSION = "REGRESSION"
    UNKNOWN = "UNKNOWN"