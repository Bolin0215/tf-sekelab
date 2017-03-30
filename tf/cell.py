import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear
from tf.nn import softmax_sum
from tf.utils import flatten, reconstruct

class AttentionCell(RNNCell):
    def __init__(self, cell, memory, mask=None, input_keep_prob=1.0):
        '''
        :param cell:
        :param memory: [N, M, J, d]
        :param mask: [N, M, J]
        :param input_keep_prob:
        '''
        self._cell = cell
        self._memory = memory
        self._mask = mask
        self._flat_memory = flatten(memory, 2)
        self._flat_mask = flatten(mask, 1)
        self._controller = AttentionCell.get_att_weights(True, input_keep_prob)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "attention_cell"):
            memory_logits = self._controller(inputs, state, self._flat_memory) #[N, M]
            sel_mem = softmax_sum(self._flat_memory, memory_logits, mask=self._flat_mask)
            return self._cell(sel_mem, state)

    @staticmethod
    def get_att_weights(bias, input_keep_prob):
        def att_weights(inputs, state, memory):
            '''
            :param inputs: [N, i]
            :param state: [N, d]
            :param memory: [N, J, i]
            :return: [N, J]
            '''
            rank = len(memory.get_shape())
            memory_size = tf.shape(memory)[rank-2]
            tiled_inputs = tf.tile(tf.expand_dims(inputs, 1), [1, memory_size, 1])
            if isinstance(state, tuple):
                tiled_state = [tf.tile(tf.expand_dims(each, 1), [1, memory_size, 1]) for each in state]
            else:
                tiled_state = [tf.tile(tf.expand_dims(state, 1), [1, memory_size, 1])]

            in_ = tf.concat([tiled_inputs] + tiled_state + [memory], 2)
            flat_in = flatten(in_, 1)
            flat_in = [tf.nn.dropout(flat_in, input_keep_prob)]
            flat_out = _linear(flat_in, 1, bias)

            out = reconstruct(flat_out, in_, 1)
            out = tf.squeeze(out, [len(in_.get_shape().as_list())-1])
            return out
        return att_weights
