import enum


class ModelType(enum.Enum):
    HEATMAP = "HEATMAP"
    REGRESSION = "REGRESSION"
    TWO_HEAD = "TWO_HEAD"
