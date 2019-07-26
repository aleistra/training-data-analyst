import shutil
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

BUCKET = None  # set from task.py
PATTERN = "of" # gets all files

# Determine CSV and label columns
CSV_COLUMNS = 'weight_pounds,is_male,mother_age,mother_race,father_race,cigarette_use,mother_married,ever_born,plurality,weight_gain_pounds,gestation_weeks'.split(',')
LABEL_COLUMN = 'weight_pounds'

# Set default values for each CSV column
CSV_DEFAULTS = [[0.0], ['Unknown'], [0], ['0'], ['0'], ['False'], ['True'], ['1'], ['Single(1)'], [30], [0]]

# Define some hyperparameters
TRAIN_STEPS = 10000
EVAL_STEPS = None
BATCH_SIZE = 512
NEMBEDS = 3
NNSIZE = [64, 16, 4]
#NTREES = 100
#MAXDEPTH = 6


def decode_csv(line_of_text):
    fields = tf.decode_csv(records = line_of_text, record_defaults = CSV_DEFAULTS, na_value='None')
    features = dict(zip(CSV_COLUMNS, fields))
    features['mother_race'] = tf.cast(features['mother_race'], 'string')
    features['father_race'] = tf.cast(features['father_race'], 'string')
    features['plurality'] = tf.cast(features['plurality'], 'string')
    features['weight_gain_pounds'] = tf.cast(features['weight_gain_pounds'], 'int32')
    if (features['weight_gain_pounds'] == 99):
        features['weight_gain_pounds'] = 30
    label = features.pop(LABEL_COLUMN) # remove label from features and store
    return features, label

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def read_dataset(filename_pattern, mode, batch_size = 512):
    def _input_fn():
    
        path_to_files = 'gs://qwiklabs-gcp-636667ae83e902b6_al/babyweight/preproc/'
        # Create list of files that match pattern.  Does support internal wildcarding e.g. "babyweight*.csv"
        file_list = tf.gfile.Glob(path_to_files + filename_pattern + "*" + PATTERN + "*")

        print(filename_pattern)
        print(file_list)
        # Create dataset from file list
        dataset = tf.data.TextLineDataset(filenames = file_list).skip(count = 1)
        dataset = dataset.map(map_func = decode_csv)

        # In training mode, shuffle the dataset and repeat indefinitely
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
            num_epochs = None 
        else:
            num_epochs = 1 

        dataset = dataset.repeat(count = num_epochs).batch(batch_size = batch_size)
        return dataset

        # This will now return batches of features, label
        return dataset
    return _input_fn

# Define feature columns
def get_categorical(name, values):
    return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(key=name, vocabulary_list=values))

    
num_cols = ['mother_age', 'gestation_weeks', 'weight_gain_pounds']
cat_cols = ['is_male', 'mother_race', 'father_race', 'cigarette_use', 'mother_married', 'plurality', 'ever_born']


cat_vocab = {
            'is_male': ['True', 'False', 'Unknown'], 
             'cigarette_use': ['True', 'False', 'None'], 
             'mother_married': ['True', 'False'], 
             'mother_race': ['1', '7', '2', '0', '3', '18', '28', '5', '48', '4', '68', '9', '78',
        '6', '38', '58'], 
             'father_race': ['1', '7', '2', '0', '3', '18', '28', '5', '48', '4', '68', '9', '78',
        '6', '38', '58'], 
             'plurality': ['Single(1)', 'Twins(2)', 'Multiple(2+)', 'Triplets(3)',
       'Quintuplets(5)', 'Quadruplets(4)'] ,
              'ever_born': ['1', '2', '3', '4', '5']
            }

def get_cols(num_cols, cat_cols, cat_vocab):
    all_cols = []
    for col in num_cols:
        all_cols.append(tf.feature_column.numeric_column(key = col))
    for col in cat_cols:
        all_cols.append(get_categorical(col, cat_vocab[col]))

    #fc_crossed_race = tf.feature_column.crossed_column(keys = ['mother_race', 'father_race'], hash_bucket_size = 100)
    
    #all_cols.append(tf.feature_column.indicator_column(categorical_column = fc_crossed_race))
    return all_cols


# Create serving input function to be able to serve predictions later using provided inputs
def serving_input_fn():
    num_placeholders = {col: tf.placeholder(dtype=tf.float32, shape=[None], name=col) for col in num_cols}     
    cat_placeholders = {col: tf.placeholder(dtype=tf.string, shape=[None], name=col) for col in cat_cols}
    
    feature_placeholders = {**num_placeholders, **cat_placeholders}
    
    features = {
        key: tf.expand_dims(input = tensor, axis = -1)
        for key, tensor in feature_placeholders.items()
    }
    
    return tf.estimator.export.ServingInputReceiver(features = features, receiver_tensors = feature_placeholders)

# create metric for hyperparameter tuning
def my_rmse(labels, predictions):
    pred_values = predictions["predictions"]
    return {"rmse": tf.metrics.root_mean_squared_error(labels = labels, predictions = pred_values)}

# Create estimator to train and evaluate
def train_and_evaluate_dnn(output_dir):
    EVAL_INTERVAL = 300
    run_config = tf.estimator.RunConfig(
        save_checkpoints_secs = EVAL_INTERVAL,
        tf_random_seed=42,
        keep_checkpoint_max = 3)

    estimator = tf.estimator.DNNRegressor(model_dir=output_dir,
                                         feature_columns = get_cols(num_cols, cat_cols, cat_vocab),
                                         hidden_units = [64,32],
                                         config=run_config)
    
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    train_spec = tf.estimator.TrainSpec(input_fn = read_dataset("train", mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = TRAIN_STEPS)
    
    exporter = tf.estimator.LatestExporter(name = "exporter", serving_input_receiver_fn = serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn = read_dataset("eval", mode=tf.estimator.ModeKeys.EVAL), exporters=exporter)
        
    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
    
def train_and_evaluate_gbt(output_dir):
    EVAL_INTERVAL = 300
    run_config = tf.estimator.RunConfig(
        save_checkpoints_secs = EVAL_INTERVAL,
        tf_random_seed=42,
        keep_checkpoint_max = 3)

    estimator = tf.estimator.BoostedTreesRegressor(model_dir=output_dir,
                                                   n_batches_per_layer = 1,
                                         feature_columns = get_cols(num_cols, cat_cols, cat_vocab),
                                         n_trees=NTREES,
                                         max_depth=MAXDEPTH,   
                                         learning_rate=0.05,          
                                         config=run_config)
    
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    train_spec = tf.estimator.TrainSpec(input_fn = read_dataset("train", mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = TRAIN_STEPS)
    
    exporter = tf.estimator.BestExporter(name = "exporter", serving_input_receiver_fn = serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn = read_dataset("eval", mode=tf.estimator.ModeKeys.EVAL), exporters=exporter)
                  
    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)    
    
