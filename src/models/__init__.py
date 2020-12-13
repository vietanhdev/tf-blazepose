from .blazepose_legacy import BlazePose as BlazePoseLegacy
from .blazepose_full import BlazePose as BlazePoseFull
from .blazepose_all_linear import BlazePose as BlazePoseAllLinear
from .blazepose_with_pushup_classify import BlazePose as BlazePoseWithClassify
from .pushup_recognition import PushUpRecognition

class ModelCreator():

    @staticmethod
    def create_model(model_name, n_points=0):

        if model_name == "SIGMOID_HEATMAP_SIGMOID_REGRESS_TWO_HEAD":
            return BlazePoseLegacy(n_points).build_model("TWO_HEAD")
        elif model_name == "SIGMOID_HEATMAP_SIGMOID_REGRESS_HEATMAP":
            return BlazePoseLegacy(n_points).build_model("HEATMAP")
        elif model_name == "SIGMOID_HEATMAP_SIGMOID_REGRESS_REGRESSION":
            return BlazePoseLegacy(n_points).build_model("REGRESSION")

        elif model_name == "SIGMOID_HEATMAP_LINEAR_REGRESS_TWO_HEAD":
            return BlazePoseFull(n_points).build_model("TWO_HEAD")
        elif model_name == "SIGMOID_HEATMAP_LINEAR_REGRESS_HEATMAP":
            return BlazePoseFull(n_points).build_model("HEATMAP")
        elif model_name == "SIGMOID_HEATMAP_LINEAR_REGRESS_REGRESSION":
            return BlazePoseFull(n_points).build_model("REGRESSION")

        elif model_name == "ALL_LINEAR_TWO_HEAD":
            return BlazePoseAllLinear(n_points).build_model("TWO_HEAD")
        elif model_name == "ALL_LINEAR_HEATMAP":
            return BlazePoseAllLinear(n_points).build_model("HEATMAP")
        elif model_name == "ALL_LINEAR_REGRESSION":
            return BlazePoseAllLinear(n_points).build_model("REGRESSION")

        elif model_name == "PUSHUP_RECOGNITION":
            return PushUpRecognition.build_model()

        elif model_name == "BLAZEPOSE_WITH_PUSHUP_CLASSIFY":
            return BlazePoseWithClassify(n_points).build_model("TWO_HEAD")
