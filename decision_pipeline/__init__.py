import warnings

warnings.filterwarnings("ignore", message="X has feature names")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from decision_pipeline.core import pipeline

__all__ = ["pipeline"]
