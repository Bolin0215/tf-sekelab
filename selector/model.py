import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper
from tf.nn import bidirectional_dynamic_rnn, linear_logits, dynamic_rnn
from tf.cell import AttentionCell

class Model:
    def __init__(self, args, TextData):
        self.args = args
        self.vocabSize = TextData.vocabSize
        self.word_emb = TextData.word_emb
        self.no_train_size = len(self.word_emb)

        #Define inputs here
        self.p = tf.placeholder('int32', [None, None, None], name='p') #batch_size, max_sent_size, max_word_num_p
        self.p_mask = tf.placeholder('bool', [None, None, None], name='p_mask')
        self.q = tf.placeholder('int32', [None, None, None], name='q') #batch_size, max_sent_size, max_word_num_q
        self.q_mask = tf.placeholder('bool', [None, None, None], name='q_mask')
        self.y = tf.placeholder('int32', [None], name='y') #batch_size, max_sent_size
        self.y_mask = tf.placeholder('bool', [None, None], name='y_mask')
        self.dropoutRate = tf.placeholder('float', None, name='dropoutRate')


        self.tensor_dict = {}

        self._build_forward()
        self._build_loss()
        self._build_acc()

        self.summary = tf.summary.merge_all()

    def _build_forward(self):

        with tf.variable_scope('emb'):
            with tf.variable_scope('emb_var'), tf.device('/cpu:0'):
                update_size = self.vocabSize
                if self.args.preTrainEmbed:
                    update_size = self.vocabSize - self.no_train_size
                word_emb_mat = tf.get_variable('word_emb_mat', dtype='float',
                                               initializer=tf.truncated_normal([update_size, self.args.embeddingSize], stddev=0.5))
                if self.args.preTrainEmbed:
                    no_train_emb = tf.Variable(self.word_emb, name='no_train_emb', trainable=False)
                    word_emb_mat = tf.concat([no_train_emb, word_emb_mat], 0)
            with tf.name_scope('word'):
                emb_p = tf.nn.embedding_lookup(word_emb_mat, self.p) #[N, M, P, d]
                emb_q = tf.nn.embedding_lookup(word_emb_mat, self.q) #[N, M, Q, d]

        cell = BasicLSTMCell(self.args.hiddenSize, state_is_tuple=True)
        d_cell = DropoutWrapper(cell, input_keep_prob=self.dropoutRate)
        p_len = tf.reduce_sum(tf.cast(self.p_mask, 'int32'), 2) #[N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 2) #[N, M]

        with tf.name_scope('Preprocess'), tf.variable_scope('lstm'):
            (fw_p, bw_p), _ = bidirectional_dynamic_rnn(d_cell, d_cell, emb_p, p_len, dtype='float', scope='p')
            Hp = tf.concat([fw_p, bw_p], 3) #[N, M, P, 2d]
            if self.args.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()
                (fw_q, bw_q), _ = bidirectional_dynamic_rnn(d_cell, d_cell, emb_q, q_len, dtype='float', scope='p')
                Hq = tf.concat([fw_q, bw_q], 3) #[N, M, Q, 2d]
            else:
                (fw_q, bw_q), _ = bidirectional_dynamic_rnn(d_cell, d_cell, emb_q, q_len, dtype='float', scope='q')
                Hq = tf.concat([fw_q, bw_q], 3)  # [N, M, Q, 2d]

        with tf.name_scope('Attention'), tf.variable_scope('attention'):
            att_cell_q = AttentionCell(cell, Hq, mask=self.q_mask, input_keep_prob=self.dropoutRate)
            Hr_p, _ = dynamic_rnn(att_cell_q, Hp, p_len, dtype='float', scope='p2q')
            att_cell_p = AttentionCell(cell, Hp, mask=self.p_mask, input_keep_prob=self.dropoutRate)
            Hr_q, _ = dynamic_rnn(att_cell_p, Hq, q_len, dtype='float', scope='q2p')

        with tf.name_scope('Aggregation'), tf.variable_scope('aggregate'):
            fw_p, _ = dynamic_rnn(d_cell, Hr_p, p_len, dtype='float', scope='agg_p')
            Agg_p = fw_p[:,:,-1]
            fw_q, _ = dynamic_rnn(d_cell, Hr_q, q_len, dtype='float', scope='agg_q')
            Agg_q = fw_q[:,:,-1]
            Agg = tf.concat([Agg_p, Agg_q], 2)

        with tf.variable_scope('out'):
            logits = linear_logits(Agg, 1, True, mask=self.y_mask, scope='logit') #[N, M]
            yp = tf.nn.softmax(logits)
            self.yp = tf.argmax(yp, axis=1, name='predict')
            self.logits = logits


    def _build_loss(self):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        self.loss = tf.reduce_mean(losses)
        tf.summary.scalar(self.loss.op.name, self.loss)
        opt = tf.train.AdamOptimizer(self.args.lrate)
        self.optOp = opt.minimize(self.loss)

    def _build_acc(self):
        correct_pred = tf.equal(tf.cast(self.yp, 'int32'), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, 'float'))
        tf.summary.scalar('acc', self.acc)

    def step(self, batch, test=False):
        feed_dict = {}
        ops = None

        feed_dict[self.p] = batch.p
        feed_dict[self.p_mask] = batch.p_mask
        feed_dict[self.q] = batch.q
        feed_dict[self.q_mask] = batch.q_mask
        feed_dict[self.y_mask] = batch.y_mask
        if not test:
            feed_dict[self.y] = batch.y

        if not test:
            feed_dict[self.dropoutRate] = self.args.dropoutRate
            ops = (self.summary, self.optOp, self.loss, self.yp)
        else:
            feed_dict[self.dropoutRate] = 1.0
            ops = self.yp

        return ops, feed_dict
