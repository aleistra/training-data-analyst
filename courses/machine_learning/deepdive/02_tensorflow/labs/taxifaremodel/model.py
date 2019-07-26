# TODO: Your code goes here. Import the necessary libraries (e.g. tensorflow, etc)

import tensorflow as tf
import shutil

CSV_COLUMN_NAMES = ["fare_amount","dayofweek","hourofday","pickuplon","pickuplat","dropofflon","dropofflat"]
CSV_DEFAULTS = [[0.0],[1],[0],[-74.0], [40.0], [-74.0], [40.7]]
FEATURE_NAMES = CSV_COLUMN_NAMES[1:]

def parse_row(row):
    fields = tf.decode_csv(records = row, record_defaults = CSV_DEFAULTS)
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    label = features.pop("fare_amount")
    return features, label

def read_dataset(csv_path):
    dataset = tf.data.Dataset.list_files(file_pattern = csv_path)
    dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filenames = filename).skip(count = 1))
    dataset = dataset.map(map_func = parse_row)
    return dataset

def train_input_fn(csv_path, batch_size = 128):
    dataset = read_dataset(csv_path)
    dataset = dataset.shuffle(buffer_size = 1000).repeat(count = None).batch(batch_size = batch_size)
    return dataset

def eval_input_fn(csv_path, batch_size = 128):
    dataset = read_dataset(csv_path)
    dataset = dataset.batch(batch_size = batch_size)
    return dataset

def serving_input_receiver_fn():
    receiver_tensors = {"dayofweek": tf.placeholder(shape=None, dtype=tf.int8),
                        "hourofday": tf.placeholder(shape=None, dtype=tf.int8),
                        "pickuplon": tf.placeholder(shape=None, dtype=tf.float32),
                        "pickuplat": tf.placeholder(shape=None, dtype=tf.float32),
                        "dropofflon": tf.placeholder(shape=None, dtype=tf.float32),
                        "dropofflat": tf.placeholder(shape=None, dtype=tf.float32)
                       }
    features = receiver_tensors
    return tf.estimator.export.ServingInputReceiver(features = features, receiver_tensors = receiver_tensors)  
    
def my_rmse(labels, predictions):
    pred_values = tf.squeeze(input = predictions["predictions"], axis = -1)
    return {
        "rmse": tf.metrics.root_mean_squared_error(labels, pred_values)
    }

def create_model(model_dir, train_steps):
    feature_cols = [tf.feature_column.numeric_column(key = k) for k in FEATURE_NAMES]
    model = tf.estimator.DNNRegressor(
    hidden_units = [10,10], # specify neural architecture
    feature_columns = feature_cols, 
    config = tf.estimator.RunConfig(
        model_dir = model_dir,
        tf_random_seed = 1,
        save_checkpoints_steps = 100
    )
)
    return model

def train_and_evaluate(params):
    OUTDIR = params["output_dir"]
    TRAIN_DATA_PATH = params["train_data_path"]
    EVAL_DATA_PATH = params["eval_data_path"]
    TRAIN_STEPS = params["train_steps"]

    model = create_model(OUTDIR, TRAIN_STEPS) 

    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: train_input_fn(TRAIN_DATA_PATH),
        max_steps = TRAIN_STEPS
    )
    
    exporter = tf.estimator.BestExporter("exporter", lambda: serving_input_receiver_fn())

    eval_spec = tf.estimator.EvalSpec(lambda: eval_input_fn(EVAL_DATA_PATH),
                                 exporters=exporter,
                                 start_delay_secs=1, 
                                 throttle_secs=1)

    tf.logging.set_verbosity(tf.logging.INFO) 
    shutil.rmtree(path = OUTDIR, ignore_errors = True)

    tf.estimator.train_and_evaluate(estimator = model, train_spec = train_spec, eval_spec = eval_spec)
