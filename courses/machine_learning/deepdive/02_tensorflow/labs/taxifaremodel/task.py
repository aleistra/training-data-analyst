import argparse
import json
import os

from . import model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--train_data_path",
        help = "GCS or local path to training data",
        required = True
    )
    parser.add_argument(
        "--train_steps",
        help = "Steps to run the training job for (default: 1000)",
        type = int,
        default = 1000
    )
    parser.add_argument(
        "--eval_data_path",
        help = "GCS or local path to evaluation data",
        required = True
    )
    parser.add_argument(
        "--output_dir",
        help = "GCS or local path to output directory",
        required = True
    )
    parser.add_argument(
        "--job-dir",
        help = "GCS path to upload package to",
        required = False
    )
    args = parser.parse_args().__dict__

    model.train_and_evaluate(args)
