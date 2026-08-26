import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

from src.core.regression_benchmark import (
    run_models,
    print_global_results,
    save_global_results_to_csv
)


if __name__ == "__main__":
    results = run_models()
    print_global_results(results)
    save_global_results_to_csv(results)
    print("\n")
