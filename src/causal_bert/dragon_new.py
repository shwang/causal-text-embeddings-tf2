import numpy as np
import tensorflow as tf
from tf_official.nlp.bert_models import pretrain_model
from tf_official.nlp import bert_modeling as modeling


def our_get_dragon_heads(include_aux: bool = True):
    """
    Changes from original dragon heads:
    * accept t as input.
    * q0 takes concat([z, t, t_aux]) as input, not z as input.
        * t_aux can be "empty".
    * q1 is always np.nan.
    """

    def dragon_heads(z: tf.Tensor, t: tf.Tensor, t_aux: tf.Tensor):
        # z is approximately pooled output from transformer.
        # We add t.
        # imp = tf.concat([z, t, t_aux], axis=1)
        batch_size = tf.shape(z)[0]
        if include_aux:
            maybe_aux = [t_aux]
        else:
            maybe_aux = []
        imp = tf.concat([z, t, *maybe_aux], axis=1)
        imp0 = tf.concat([z, tf.zeros([batch_size, 1]), *maybe_aux], axis=1)
        imp1 = tf.concat([z, tf.ones([batch_size, 1]), *maybe_aux], axis=1)
        # imp = t
        # breakpoint()
        # print(imp)
        # print(imp0)
        # print(imp1)

        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(200, activation='relu'),
             tf.keras.layers.Dense(200, activation='relu'),
             tf.keras.layers.Dense(13, activation='softmax'),
             ],
            name="q",
        )

        q_logits = model(imp)  # name: q
        q0_logits = model(imp0)
        q1_logits = model(imp1)
        #  def rename(x, name: str):
        #      lamb = tf.keras.layers.Lambda(lambda t: t, name=name)
        #      return lamb(x)

        # Naming (for keras output references later) is unfortunate.
        #   I tried a few ways to rename these outputs, including name
        #   scopes to no avail.
        q_pred = tf.argmax(q_logits, axis=1, name="q_pred")  # name: tf.math.argmax
        q0_pred = tf.argmax(q0_logits, axis=1, name="q0_pred")  # tf.math.argmax_1
        q1_pred = tf.argmax(q1_logits, axis=1, name="q1_pred")  # tf.math.argmax_2
        # q_pred_onehot = tf.one_hot(q_pred, depth=13)

        g = tf.keras.layers.Dense(200, activation='relu')(z)
        g = tf.keras.layers.Dense(200, activation='relu')(g)
        g = tf.keras.layers.Dense(1, activation='sigmoid', name='g')(g)
        # note: They only have a linear+sigmoid network for g.
        # Maybe adding more layers will help.

        return (g, q_logits, q_pred, q0_pred, q1_pred,
                q_logits, q0_logits, q1_logits)

    return dragon_heads


def dragon_model_ours(bert_config,
                      max_seq_length: int,
                      binary_outcome: bool = False,
                      use_unsup=False,
                      max_predictions_per_seq=20,
                      unsup_scale=1.,
                      include_aux=True,
                      ):
    assert binary_outcome is False
    if use_unsup:
        pt_model, bert_model = pretrain_model(bert_config,
                                              max_seq_length,
                                              max_predictions_per_seq,
                                              initializer=None)

        inputs = pt_model.input
        unsup_loss = pt_model.outputs
        unsup_loss = unsup_scale * tf.reduce_mean(unsup_loss)
    else:
        input_word_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

        inputs = {
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids,
        }

        bert_model = modeling.get_bert_model(
            input_word_ids,
            input_mask,
            input_type_ids,
            config=bert_config)

        unsup_loss = lambda: 0  # tf.convert_to_tensor(0.) doesn't work...


    treatment = tf.keras.layers.Input(
        shape=(1,), dtype=tf.float32, name='treatment')

    more_inputs = {
        'treatment': treatment,
        # TODO(shwang): When adding more_treatment later, also
        #   use t_aux in `our_get_dragon_heads()`
    }
    if include_aux:
        more_treatment = tf.keras.layers.Input(
            shape=(9,), dtype=tf.float32, name='more_treatment')
        more_inputs['more_treatment'] = more_treatment
    else:
        more_treatment = None
    inputs.update(**more_inputs)

    pooled_output = bert_model.outputs[0]

    head_model = our_get_dragon_heads(include_aux=include_aux)
    g, q, q_pred, q0_pred, q1_pred, q_logits, \
        q0_logits, q1_logits = head_model(pooled_output, treatment, more_treatment)

    # Oh interesting, we could insert output here.
    # output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
    #     pooled_output)

    dragon_model = tf.keras.Model(
        inputs=inputs,
        outputs=[g, q, q_pred, q0_pred, q1_pred, q_logits, q0_logits, q1_logits])
    dragon_model.add_loss(unsup_loss)

    return dragon_model, bert_model
