"""
Simple pre-processing for PeerRead papers.
Takes in JSON formatted data from ScienceParse and outputs a tfrecord


Reference example:
https://github.com/tensorlayer/tensorlayer/blob/9528da50dfcaf9f0f81fba9453e488a1e6c8ee8f/examples/data_process/tutorial_tfrecord3.py
"""

import argparse
import os
import pathlib
import pandas as pd
import random

import tensorflow as tf
import tqdm
import numpy as np
import bert.tokenization as tokenization
import reddit.data_cleaning.reddit_posts as rp

rng = random.Random(0)


def process_row_record(row_dict: dict, tokenizer):
    # Fixes https://github.com/google-research/bert/issues/1133
    import sys
    import absl.flags
    sys.argv = ["preserve_unused_tokens=False"]
    absl.flags.FLAGS(sys.argv)
    absl.flags.FLAGS(["preserve_unused_tokens=False"])

    # IDEA: Save with text (len < 1) and (len < 10) and (len < 20)

    text = row_dict['sentence_deleted_hedge']
    if len(text) < 1:  # TODO(shwang): Skip <10 in the future
        """ e.g. len(text) < 10
        Warning: Skipping text=178-80). due to short length.
        Warning: Skipping text=. due to short length.
        Warning: Skipping text=B. due to short length.
        Warning: Skipping text=2003). due to short length.
        Warning: Skipping text=. due to short length.
        Warning: Skipping text=12, pp. due to short length.
        Warning: Skipping text=. due to short length.
        Warning: Skipping text=63, no. due to short length.
        Warning: Exceeded maximum sequence length. Truncating from 288 to 256 tokens.
        Warning: Skipping text=(2002). due to short length.
        Warning: Skipping text=Proof. due to short length.
        Warning: Skipping text=7. due to short length.
        """
        # print("Warning: Skipping text={} due to short length.".format(text))
        return None, None
    tokens = tokenizer.tokenize(text)

    text_features = {'sentence_deleted_hedge_tokens': tokens}
    context_features = row_dict
    return text_features, context_features


def bert_process_sentence(example_tokens, max_seq_length, tokenizer, segment=1):
    """Tokenization and pre-processing of text as expected by Bert"""
    # Account for [CLS] and [SEP] with "- 2"
    if len(example_tokens) > max_seq_length - 2:
        print("Warning: Exceeded maximum sequence length. Truncating from {} to {} tokens.".format(
            len(example_tokens) + 2,
            max_seq_length,
        ))
        print("Original tokens: {}".format(" ".join(example_tokens)))
        example_tokens = example_tokens[0:(max_seq_length - 2)]

    # The convention in BERT for single sequences is:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence.

    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(segment)
    for tidx, token in enumerate(example_tokens):
        tokens.append(token)
        segment_ids.append(segment)

    tokens.append("[SEP]")
    segment_ids.append(segment)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def reddit_to_bert_Example(text_features, context_features, max_seq_length, tokenizer):
    """
    Parses the input paper into a tf.Example as expected by Bert
    Note: the docs for tensorflow Example are awful ¯\_(ツ)_/¯
    """
    features = {}

    tokens, padding_mask, segments = \
        bert_process_sentence(text_features['sentence_deleted_hedge_tokens'], max_seq_length, tokenizer)

    features["token_ids"] = _int64_feature(tokens)
    features["token_mask"] = _int64_feature(padding_mask)
    features["segment_ids"] = _int64_feature(segments)

    # non-sequential features from context_dict.
    # Note that we drop all string features (keep only int or float).
    tf_context_features, tf_context_features_types = _dict_of_nonlist_numerical_to_tf_features(context_features)
    features = {**tf_context_features, **features}
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto,
    e.g, An integer label.
    """
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Wrapper for inserting a float Feature into a SequenceExample proto,
    e.g, An integer label.
    """
    if isinstance(value, list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto,
    e.g, an image in byte
    """
    # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _dict_of_nonlist_numerical_to_tf_features(my_dict):
    """
    Strip out non-numerical features
    Returns tf_features_dict: a dictionary suitable for passing to tf.train.example
            tf_types_dict: a dictionary of the tf types of previous dict

    """

    tf_types_dict = {}
    tf_features_dict = {}
    for k, v in my_dict.items():
        if isinstance(v, int) or isinstance(v, bool):
            tf_features_dict[k] = _int64_feature(v)
            tf_types_dict[k] = tf.int64
        elif isinstance(v, float):
            tf_features_dict[k] = _float_feature(v)
            tf_types_dict[k] = tf.float32
        else:
            pass

    return tf_features_dict, tf_types_dict


def process_reddit_dataset(data_dir, out_dir, out_file, max_abs_len, tokenizer, subsample, use_latest_reddit):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # TODO(shwang): Replace with df.read_csv I think.
    df = pd.read_csv("../our_data/sentences_shwang_all.csv")
    #     if data_dir:
    #         reddit_df = rp.load_reddit(path=data_dir, use_latest=use_latest_reddit, convert_columns=True)
    #     else:
    #         reddit_df = rp.load_reddit(use_latest=use_latest_reddit, convert_columns=True)

    # add persistent record of the index of the data examples
    df['index'] = df.index
    records = df.to_dict('records')
    FAST = True
    if FAST:
        records = records[::1000]
    n_records = len(records)
    print("Hello, I prioritize my scalp, and I loaded n={} records.".format(n_records))
    print("The columns that I loaded are: {}".format(df.columns))

    # random_example_indices = np.arange(len(records))
    # np.random.shuffle(random_example_indices)
    # random_response_mask = np.random.randint(0, 2, len(records))

    out_path = pathlib.Path(out_dir, out_file)
    print("Writing to {}".format(out_path))

    with tf.io.TFRecordWriter(out_dir + "/" + out_file) as writer:
        for idx, row_dict in enumerate(tqdm.tqdm(records, desc="Records")):
            if subsample and idx >= subsample:
                break

            text_features, context_features = process_row_record(row_dict, tokenizer)
            # turn it into a tf.data example
            if text_features is not None:
                many_split = rng.randint(0, 100)  # useful for easy data splitting later
                extra_context = {'many_split': many_split}
                context_features.update(extra_context)
                row_ex = reddit_to_bert_Example(text_features, context_features,
                                                max_seq_length=max_abs_len,
                                                tokenizer=tokenizer)
                feat_keys = row_ex.features.ListFields()[0][1].keys()
                assert "male_ratio" in feat_keys
                assert "token_ids" in feat_keys
                writer.write(row_ex.SerializeToString())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None)
    # parser.add_argument('--out-dir', type=str, default='../tmp/confidence')
    parser.add_argument('--out-dir', type=str, default='../dat/shwang_tiny')
    parser.add_argument('--out-file', type=str, default='proc.tf_record')
    parser.add_argument('--vocab-file', type=str, default='../pre-trained/uncased_L-12_H-768_A-12/vocab.txt')
    parser.add_argument('--max-abs-len', type=int, default=512)
    parser.add_argument('--subsample', type=int, default=0)
    parser.add_argument('--use-latest-reddit', type=bool, default=True)

    args = parser.parse_args()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=True)

    process_reddit_dataset(args.data_dir, args.out_dir, args.out_file,
                           args.max_abs_len, tokenizer, args.subsample, args.use_latest_reddit)


if __name__ == "__main__":
    main()
