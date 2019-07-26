import argparse
import json
import os

from . import model

import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket",
        help = "GCS path to data. We assume that data is in \
        gs://BUCKET/babyweight/preproc/",
        required = True
    )
    parser.add_argument(
        "--output_dir",
        help = "GCS location to write checkpoints and export models",
        required = True
    )
    parser.add_argument(
        "--batch_size",
        help = "Number of examples to compute gradient over.",
        type = int,
        default = 512
    )
    parser.add_argument(
        "--job-dir",
        help = "this model ignores this field, but it is required by gcloud",
        default = "junk"
    )
    
    parser.add_argument(
        "--nnsize",
        help = "Hidden layer sizes to use for DNN (string, comma-separated)",
        default="[10,10]"
    )
    
    parser.add_argument(
        "--nembeds",
        help = "Embedding size of a cross of n key parameters - this will be a small integer",
        default = 3)
    
    parser.add_argument(
        "--ntrees",
        help = "Number of trees",
        default = 100)
    
    parser.add_argument(
        "--maxdepth",
        help = "Depth of trees",
        default = 6)
    
    parser.add_argument(
        "--train_examples",
        help="Number of examples (in thousands) to run the training job",
        default=1)
    
    parser.add_argument(
        "--eval_steps",
        help="Steps for which to evaluate model",
        default=100)
    
    parser.add_argument(
        "--pattern",
        help = "Pattern that appears in filename",
        default = "of")
        
    # Parse arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # Pop unnecessary args needed for gcloud
    arguments.pop("job-dir", None)

    # Assign the arguments to the model variables
    output_dir = arguments.pop("output_dir")
    model.BUCKET     = arguments.pop("bucket")
    model.BATCH_SIZE = int(arguments.pop("batch_size"))
    model.TRAIN_STEPS = 200000 
    #model.TRAIN_STEPS = (int(arguments.pop("train_examples") * 1000)) / model.BATCH_SIZE
    model.EVAL_STEPS = arguments.pop("eval_steps")    

    #print ("Will train for {} steps using batch_size={}".format(model.TRAIN_STEPS, model.BATCH_SIZE))
    model.PATTERN = arguments.pop("pattern")
    model.NEMBEDS= arguments.pop("nembeds")
    model.NNSIZE = arguments.pop("nnsize")
    #print ("Will use DNN size of {}".format(model.NNSIZE))
  
    model.MAXDEPTH = int(arguments.pop("maxdepth")) 
    model.NTREES = int(arguments.pop("ntrees"))
    
    print ("Will train on {} trees with max depth of {}".format(model.NTREES, model.MAXDEPTH))
    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get("TF_CONFIG", "{}")
        ).get("task", {}).get("trial", "")
    )

    # Run the training job
    model.train_and_evaluate_gbt(output_dir)
