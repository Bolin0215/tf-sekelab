import tensorflow as tf
from tf.utils import flatten, reconstruct, exp_mask
from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear

def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None, swap_memory=False,
                              time_major=False, scope=None):
    flat_inputs = flatten(inputs, 2)
    flat_len = flatten(sequence_length, 0)
    (flat_fw_outputs, flat_bw_outputs), final_state = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len, initial_state_bw=initial_state_bw,
                                        initial_state_fw=initial_state_fw, dtype=dtype, parallel_iterations=parallel_iterations,
                                        swap_memory=swap_memory, time_major=time_major, scope=scope)

    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)

    return (fw_outputs, bw_outputs), final_state

def dynamic_rnn(cell, inputs, sequence_length=None,
                              initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False,
                              time_major=False, scope=None):
    flat_inputs = flatten(inputs, 2)
    flat_len = flatten(sequence_length, 0)
    flat_outputs, final_state = \
        tf.nn.dynamic_rnn(cell, flat_inputs, sequence_length=flat_len, initial_state=initial_state,
                                        dtype=dtype, parallel_iterations=parallel_iterations,
                                        swap_memory=swap_memory, time_major=time_major, scope=scope)

    outputs = reconstruct(flat_outputs, inputs, 2)

    return outputs, final_state

def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or 'softmax'):
        if mask is not None:
            logits = exp_mask(logits, mask)
        flat_logits = flatten(logits, 1)
        flat_out = tf.nn.softmax(flat_logits)
        out = reconstruct(logits, flat_out, 1)
        return out

def softmax_sum(target, logits, mask=None, scope=None):
    with tf.name_scope(scope or 'softmax_sum'):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank-2)
        return out

def linear_logits(input, output_size, bias, mask=None, input_keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or 'linear'):
        flat_input = flatten(input, 1)
        flat_input = [tf.nn.dropout(flat_input, input_keep_prob)]
        flat_out = _linear(flat_input, output_size, bias)
        out = reconstruct(flat_out, input, 1)
        out = tf.squeeze(out, [len(input.get_shape().as_list())-1])
        if mask is not None:
            out = exp_mask(out, mask)
        return out
