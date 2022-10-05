# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BERT classification finetuning runner in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os
import pathlib
import time

import numpy as np
import pandas as pd

from absl import app
from absl import flags
import tensorflow as tf
# import tensorflow_addons as tfa
import tqdm

from tf_official.nlp import bert_modeling as modeling
from tf_official.nlp.bert import tokenization, common_flags
from tf_official.utils.misc import tpu_lib
from causal_bert import bert_models, dragon_new as our_bert_models
from causal_bert.data_utils import dataset_to_pandas_df, filter_training

from reddit.dataset.dataset import (make_input_fn_from_file, make_real_labeler, make_subreddit_based_simulated_labeler,
                                    AUX_FEATURE_NAMES)


common_flags.define_common_bert_flags()

flags.DEFINE_enum(
    'mode', 'train_and_predict', ['train_only', 'train_and_predict', 'predict_only'],
    'One of {"train_and_predict", "predict_only"}. `train_and_predict`: '
    'trains the model and make predictions. '
    '`predict_only`: loads a trained model and makes predictions.')

flags.DEFINE_string('saved_path', None,
                    'Relevant only if mode is predict_only. Path to pre-trained model')

flags.DEFINE_bool(
    "do_masking", False,
    "Whether to randomly mask input words during training (serves as a sort of regularization)")

flags.DEFINE_float("treatment_loss_weight", 1.0, "how to weight the treatment prediction term in the loss")


flags.DEFINE_bool(
    "fixed_feature_baseline", False,
    "Whether to use BERT to produced fixed features (no finetuning)")


flags.DEFINE_string('input_files', None,
                    'File path to retrieve training data for pre-training.')

# Model training specific flags.
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')

flags.DEFINE_integer('train_batch_size', 16, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 16, 'Batch size for evaluation.')
flags.DEFINE_string(
    'hub_module_url', None, 'TF-Hub path/url to Bert module. '
                            'If specified, init_checkpoint flag should not be used.')

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("seed", 0, "Seed for rng.")

# Data splitting details
flags.DEFINE_integer("num_splits", 10,
                     "number of splits")
flags.DEFINE_string("dev_splits", '9', "indices of development splits")
flags.DEFINE_string("test_splits", '9', "indices of test splits")

# Flags specifically related to PeerRead experiment

flags.DEFINE_string(
    "treatment", "male_ratio",
    "Covariate used as treatment."
)
flags.DEFINE_string("subreddits", '13,6,8', "the list of subreddits to train on")
flags.DEFINE_bool("use_subreddit", False, "whether to use the subreddit index as a feature")
flags.DEFINE_string("simulated", 'real', "whether to use real data ('real'), attribute based ('attribute'), "
                                         "or propensity score-based ('propensity') simulation"),
flags.DEFINE_float("beta0", 0.25, "param passed to simulated labeler, treatment strength")
flags.DEFINE_float("beta1", 0.0, "param passed to simulated labeler, confounding strength")
flags.DEFINE_float("gamma", 0.0, "param passed to simulated labeler, noise level")
flags.DEFINE_float("exogenous_confounding", 0.0, "amount of exogenous confounding in propensity based simulation")
flags.DEFINE_string("base_propensities_path", '', "path to .tsv file containing a 'propensity score' for each unit,"
                                                  "used for propensity score-based simulation")

flags.DEFINE_string("simulation_mode", 'simple', "simple, multiplicative, or interaction")

flags.DEFINE_string("prediction_file", "../output/predictions.tsv", "path where predictions (tsv) will be written")

FLAGS = flags.FLAGS


def make_unique_filename():
    import datetime
    import randomname
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    rand_name = randomname.get_name()
    return f"{timestamp}_{rand_name}"

Q_PRED_NAME = "tf.math.argmax"
Q_PRED_ONE_HOT_NAME = "tf.one_hot"

def _keras_format(features, labels):
    # features, labels = sample
    y = labels['outcome']
    t = labels['treatment']
    y = tf.cast(y, dtype=tf.int64)
    # one_hots = tf.one_hot(y, depth=13)
    # t = y = tf.random.normal(tf.shape(y))
    new_labels = {'g': t, 'q': y, Q_PRED_NAME: y} # , Q_PRED_ONE_HOT_NAME: one_hots}
    more_treatment_raw = [labels[key] for key in AUX_FEATURE_NAMES]
    more_treatment_raw = [tf.cast(feat, tf.float32) for feat in more_treatment_raw]
    more_treatment = tf.concat(more_treatment_raw, axis=1)
    new_features = dict(
        **features, treatment=t, more_treatment=more_treatment)
    return new_features, new_labels# , sample_weights


def make_dataset(is_training: bool, do_masking=False, force_keras_format=False):
    if FLAGS.simulated == 'real':
        # labeler = make_real_labeler(FLAGS.treatment, 'log_score')
        labeler = make_real_labeler(FLAGS.treatment, 'uncertainty_output')
    else:
        raise ValueError(FLAGS.simulated)
    # elif FLAGS.simulated == 'attribute':
    #     labeler = make_subreddit_based_simulated_labeler(FLAGS.beta0, FLAGS.beta1, FLAGS.gamma, FLAGS.simulation_mode,
    #                                                      seed=0)
    # else:
    #     raise Exception("simulated flag not recognized")

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    dev_splits = [int(s) for s in str.split(FLAGS.dev_splits)]
    test_splits = [int(s) for s in str.split(FLAGS.test_splits)]

    if FLAGS.subreddits == '':
        subreddits = None
    else:
        subreddits = [int(s) for s in FLAGS.subreddits.split(',')]

    train_input_fn = make_input_fn_from_file(
        input_files_or_glob=FLAGS.input_files,
        seq_length=FLAGS.max_seq_length,
        num_splits=FLAGS.num_splits,
        dev_splits=dev_splits,
        test_splits=test_splits,
        tokenizer=tokenizer,
        do_masking=do_masking,
        subreddits=subreddits,
        is_training=is_training,
        shuffle_buffer_size=25000,  # note: bert hardcoded this, and I'm following suit
        seed=FLAGS.seed,
        labeler=labeler,
        filter_train=is_training)

    batch_size = FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size
    dataset = train_input_fn(params={'batch_size': batch_size})

    # format expected by Keras for training
    if is_training or force_keras_format:
        # Steven: We are relying on validation set to produce some loss metrics for report.
        # Therefore, we always want to map into keras format.
        # dataset = filter_training(dataset)
        dataset = _map_keras_format(dataset)

    return dataset


def _map_keras_format(ds):
    return ds.map(_keras_format,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)


def make_dragonnet_metrics():
    Q_LABEL_METRICS = dict(
        accuracy=tf.keras.metrics.Accuracy,
        # prec=tf.keras.metrics.Precision,
        # recall=tf.keras.metrics.Recall,
        # auc=tf.keras.metrics.AUC,
    )

    # Q_PRED_ONE_HOT_METRICS = dict(
    #     f1_weighted=functools.partial(tfa.metrics.F1Score, num_classes=13, average="weighted"),
    # )

    METRICS = dict(
        xent=tf.keras.losses.SparseCategoricalCrossentropy,
        # tf.keras.metrics.Precision,
        # tf.keras.metrics.Recall,
        # tf.keras.metrics.AUC,
        # functools.partial(tfa.metrics.F1Score, num_classes=13, average="weighted"),
    )

    # NAMES = ['binary_accuracy', 'precision', 'recall', 'auc', 'f1_score_weighted']
    NAMES = ['xent']
    CONT_METRICS = [
        tf.keras.metrics.MeanSquaredError
    ]

    CONT_NAMES = ['mse']

    g_metrics = [m(name='metrics/' + n) for m, n in zip(CONT_METRICS, CONT_NAMES)]
    q_metrics = [metric(name='metrics/' + k) for k, metric in METRICS.items()]
    q_pred_metrics = [metric(name='metrics/q_pred/' + k) for k, metric in Q_LABEL_METRICS.items()]
    # q_pred_one_hot_metrics = [metric(name='metrics/q_pred/' + k) for k, metric in Q_PRED_ONE_HOT_METRICS.items()]
    return {'g': g_metrics, 'q': q_metrics,
            Q_PRED_NAME: q_pred_metrics,
            # Q_PRED_ONE_HOT_NAME: q_pred_one_hot_metrics,
            }


def main(_):
    # Users should always run this script under TF 2.x
    assert tf.version.VERSION.startswith('2.')
    tf.random.set_seed(FLAGS.seed)

    if FLAGS.model_dir:
        tf_log_root = pathlib.Path(FLAGS.model_dir)
    else:
        tf_log_root = pathlib.Path('../output/gender_uncertainty_logs/')
    #
    # Configuration stuff
    #
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    epochs = FLAGS.num_train_epochs
    # This is hard-coded here because TFRecordDataset does not store the length. =(
    # But anyways, the actual data length is 417,895. Maybe there is a discrepancy because they dropped
    # data with extremal values (vaguely remember something like this from the paper).
    # train_data_size = 11778
    # train_data_size = 90000

    train_data_size = get_tf_ds_len(FLAGS.input_files)
    print(f"Counted training {train_data_size} examples.")
    # Aww I'm so clever =) 💆
    # train_data_size = 50

    steps_per_epoch = int(train_data_size / FLAGS.train_batch_size) * 0.9
    steps_per_val_epoch = steps_per_epoch * 0.1
    warmup_steps = int(epochs * train_data_size * 0.1 / FLAGS.train_batch_size)
    initial_lr = FLAGS.learning_rate

    strategy = None
    if FLAGS.strategy_type == 'mirror':
        strategy = tf.distribute.MirroredStrategy()
    elif FLAGS.strategy_type == 'tpu':
        cluster_resolver = tpu_lib.tpu_initialize(FLAGS.tpu)
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    else:
        raise ValueError('The distribution strategy type is not supported: %s' %
                         FLAGS.strategy_type)

    #
    # Modeling and training
    #

    # the model
    def _get_dragon_model(do_masking):
        if not FLAGS.fixed_feature_baseline:
            dragon_model, core_model = (
                # bert_models.dragon_model(
                our_bert_models.dragon_model_ours(
                    bert_config,
                    max_seq_length=FLAGS.max_seq_length,
                    binary_outcome=False,
                    use_unsup=do_masking,
                    max_predictions_per_seq=20,
                    unsup_scale=1.))
        else:
            dragon_model, core_model = bert_models.derpy_dragon_baseline(
                bert_config,
                max_seq_length=FLAGS.max_seq_length,
                binary_outcome=False)

        # WARNING: the original optimizer causes a bug where loss increases after first epoch
        # dragon_model.optimizer = optimization.create_optimizer(
        #     FLAGS.train_batch_size * initial_lr, steps_per_epoch * epochs, warmup_steps)
        dragon_model.optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.train_batch_size * initial_lr)
        return dragon_model, core_model

    log_dir = tf_log_root / make_unique_filename()

    if (FLAGS.mode == 'train_and_predict') or (FLAGS.mode == 'train_only'): 
        # training. strategy.scope context allows use of multiple devices
        with strategy.scope():
            keras_train_data = make_dataset(is_training=True, do_masking=FLAGS.do_masking)
            keras_train_data.prefetch(4)

            dragon_model, core_model = _get_dragon_model(FLAGS.do_masking)
            optimizer = dragon_model.optimizer

            if FLAGS.init_checkpoint:
                checkpoint = tf.train.Checkpoint(model=core_model)
                checkpoint.restore(FLAGS.init_checkpoint).assert_existing_objects_matched()

            latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
            if latest_checkpoint:
                dragon_model.load_weights(latest_checkpoint)

            # TODO(shwang): Note that the model has no unsupervised loss!
            dragon_model.compile(
                run_eagerly=False,
                optimizer=optimizer,
                loss={
                    'g': 'mean_squared_error',
                    # 'q': 'mean_squared_error',
                    'q': tf.keras.losses.SparseCategoricalCrossentropy(),
                },
                loss_weights={
                    'g': FLAGS.treatment_loss_weight,
                    'q': 0.2,
                },
                weighted_metrics=make_dragonnet_metrics(),
            )

            summary_callback = tf.keras.callbacks.TensorBoard(log_dir, update_freq=100)
            print(f"🐸 [LOGGING] Logging to {log_dir}")
            checkpoint_dir = os.path.join(log_dir, 'model_checkpoint.{epoch:02d}')
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, save_weights_only=True, period=10)

            callbacks = [summary_callback, checkpoint_callback]

            val_data = make_dataset(is_training=False, do_masking=FLAGS.do_masking, force_keras_format=True)
            dragon_model.fit(
                x=keras_train_data,
                # validation_data=val_data,  # Where can I get my hands on this? =)
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                # validation_steps=steps_per_val_epoch,
                callbacks=callbacks)

        # save a final model checkpoint (so we can restore weights into model w/o training idiosyncracies)
        model_export_path = log_dir / 'trained/dragon.ckpt'
        model_export_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = tf.train.Checkpoint(model=dragon_model)
        saved_path = checkpoint.save(model_export_path)
        # How does this different from trained/dragon.ckpt-1
        # print(saved_path)
    else:
        saved_path = FLAGS.saved_path

    # make predictions and write to file
    if FLAGS.mode != 'train_only':
        # create data and model w/o masking
        eval_data = make_dataset(is_training=False, do_masking=False)

        eval_data_keras = _map_keras_format(eval_data)
        dragon_model, core_model = _get_dragon_model(do_masking=False)
        # reload the model weights (necessary because we've obliterated the masking)
        checkpoint = tf.train.Checkpoint(model=dragon_model)
        checkpoint.restore(saved_path).assert_existing_objects_matched()
        # loss added as simple hack to bizzarre keras bug that requires compile for predict, and a loss for compile
        dragon_model.add_loss(lambda: 0)
        dragon_model.compile()

        cached_data = []
        for batch in eval_data_keras:
            cached_data.append(batch)
        n_batches = len(cached_data)
        del cached_data

        print("🐸 Begin generating predictions.tsv!")
        outputs = dragon_model.predict(
            eval_data_keras,
            callbacks=[TQDMPredictCallback(total=n_batches)],
        )

        out_dict = {}
        out_dict['g'] = outputs[0].squeeze()
        # This is a distribution, not meant for saving.
        # out_dict['q'] = outputs[1].squeeze()
        out_dict['q'] = outputs[2].squeeze()
        out_dict['q0'] = outputs[3].squeeze()
        out_dict['q1'] = outputs[4].squeeze()
        # Okay, time to concat that stuff after.
        # out_dict2 = {k: np.concatenate(v) for k, v in out_dict.items()}
        # out_dict['q_one_hot'] = outputs[5].squeeze()

        #     # out_dict['q'].append(outputs[2].numpy().squeeze())
        #     # out_dict['q0'].append(outputs[3].numpy().squeeze())
        #     # out_dict['q1'].append(outputs[4].numpy().squeeze())
        predictions = pd.DataFrame(out_dict)
        label_dataset = eval_data.map(lambda f, l: l)
        data_df = dataset_to_pandas_df(label_dataset)

        outs = data_df.join(predictions)
        prediction_path = log_dir / "predictions.tsv"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with tf.io.gfile.GFile(prediction_path, "w") as writer:
            writer.write(outs.to_csv(sep="\t"))
        print("Wrote predictions to {}".format(prediction_path))
        
        
def silly_main(path):
    def get_ds_len(ds) -> int:
        i = 0
        for _ in ds:
            i += 1
        return i
    raw_ds = tf.data.TFRecordDataset([path])
    x = get_ds_len(raw_ds)
    print(x)
    return x


def get_tf_ds_len(path) -> int:
    def get_ds_len(ds) -> int:
        i = 0
        for _ in ds:
            i += 1
        return i
    raw_ds = tf.data.TFRecordDataset([path])
    x = get_ds_len(raw_ds)
    return x


SMALL_DS_PATH = pathlib.Path("../dat/shwang_small/proc.tf_record")
FULL_DS_PATH = pathlib.Path("../dat/shwang_full/proc.tf_record")


class TQDMPredictCallback(tf.keras.callbacks.Callback):
    def __init__(self, custom_tqdm_instance=None, tqdm_cls=tqdm.tqdm, **tqdm_params):
        super().__init__()
        self.tqdm_cls = tqdm_cls
        self.tqdm_progress = None
        self.prev_predict_batch = None
        self.custom_tqdm_instance = custom_tqdm_instance
        self.tqdm_params = tqdm_params

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        self.tqdm_progress.update(batch - self.prev_predict_batch)
        self.prev_predict_batch = batch

    def on_predict_begin(self, logs=None):
        self.prev_predict_batch = 0
        if self.custom_tqdm_instance:
            self.tqdm_progress = self.custom_tqdm_instance
            return

        # total = self.params.get('steps')
        # if total:
        #     total -= 1

        self.tqdm_progress = self.tqdm_cls(**self.tqdm_params)

    def on_predict_end(self, logs=None):
        if self.tqdm_progress is not None and not self.custom_tqdm_instance:
            self.tqdm_progress.close()


if __name__ == '__main__':
    # silly_main(SMALL_DS_PATH)
    # tf.data.experimental.enable_debug_mode()  # using tensorflow 2.5.0rc0
    flags.mark_flag_as_required('bert_config_file')
    # flags.mark_flag_as_required('input_meta_data_path')
    # flags.mark_flag_as_required('model_dir')
    app.run(main)
