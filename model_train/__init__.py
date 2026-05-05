import warnings

warnings.filterwarnings("ignore", message="X has feature names")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
