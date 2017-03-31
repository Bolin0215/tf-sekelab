import tensorflow as tf
import numpy as np
import pickle as pkl
import os
from tqdm import tqdm
import argparse
from selector.model import Model
import datetime
from sklearn.metrics import accuracy_score
from selector.textData import textData

class Tagger:
    def __init__(self):
        self.args = None
        self.textData = None
        self.model = None
        self.globalStep = 0
        self.out_file = None

    @staticmethod
    def parseArgs(args):
        parser = argparse.ArgumentParser()
        dir = os.path.join('.','data','squad')
        parser.add_argument('--emb_dir', default=dir,help="embedding vectors directory")
        parser.add_argument('--data_dir', default=dir,help='data directory')
        parser.add_argument('--lrate', type=float, default=0.02, help='learning rate')
        parser.add_argument('--embeddingSize', type=int, default=100)
        parser.add_argument('--windowSize', type=int, default=3)
        parser.add_argument('--dropoutRate', type=float, default=1.0)
        parser.add_argument('--batchSize', type=int, default=2)
        parser.add_argument('--hiddenSize', type=int, default=50)
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--mode', default='train', help='train/test')
        parser.add_argument('--load', type=bool, default=False, help='load params at test mode')
        parser.add_argument('--load_path', default=None, help='load path')
        parser.add_argument('--preTrainEmbed', type=bool, default=True)
        parser.add_argument('--device', type=str, default='/cpu:0')
        parser.add_argument('--share_lstm_weights', type=bool, default=False)
        parser.add_argument('--saveAtt', default=False, type=bool)
        parser.add_argument('--attention', type=bool, default=True)
        parser.add_argument('--globalContext', type=bool, default=False)
        parser.add_argument('--phraseEmbedding', type=bool, default=False)

        out_dir = os.path.join('.','out')
        parser.add_argument('--out_dir', default=out_dir)

        return parser.parse_args(args)

    def main(self, args=None):
        print ('TensorFlow version v{}'.format(tf.__version__))

        sessConfig = tf.ConfigProto(allow_soft_placement = True)
        sessConfig.gpu_options.allow_growth = True

        self.args = self.parseArgs(args)

        if self.args.mode == 'train':
            if not os.path.isdir(self.args.out_dir):
                os.mkdir(self.args.out_dir)
            self.out_dir = os.path.join(self.args.out_dir,
                                        str(self.args.batchSize) + '_' + str(self.args.embeddingSize) + '_' + \
                                        str(self.args.hiddenSize) + '_' + str(
                                            self.args.dropoutRate) + '_' + \
                                        str(self.args.lrate))
            if not os.path.isdir(self.out_dir):
                os.mkdir(self.out_dir)
            summary_dir = os.path.join(self.out_dir, 'summary')
            if not os.path.isdir(summary_dir):
                os.mkdir(summary_dir)
            self.args.summary_dir = summary_dir
            best_dir = os.path.join(self.out_dir, 'best')
            if not os.path.isdir(best_dir):
                os.mkdir(best_dir)
            self.args.best_dir = best_dir

            self.out_file = os.path.join(self.out_dir, 'out')
            assert not os.path.isfile(self.out_file)

            self.out_dir = os.path.join(self.out_dir, 'val')
            if not os.path.isdir(self.out_dir):
                os.mkdir(self.out_dir)


            self.textData = textData(self.args)
            self.sess = tf.Session(config=sessConfig)
            with tf.device(self.args.device):
                self.model = Model(self.args, self.textData)
                init = tf.global_variables_initializer()
                self.sess.run(init)
                self.train(self.sess)
                self.sess.close()

        elif self.args.mode == 'test':
            assert self.args.load_path
            self.textData = textData(self.args)
            self.sess = tf.Session(config=sessConfig)
            with tf.device(self.args.device):
                self.model = Model(self.args, self.textData)
                init = tf.global_variables_initializer()
                self.sess.run(init)
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(self.args.load_path)
                if ckpt and ckpt.model_checkpoint_path:
                    print ('load model from {}'.format(ckpt.model_checkpoint_path))
                    saver.restore(self.sess, tf.train.latest_checkpoint(self.args.load_path))
                self.forward(self.sess)
                self.sess.close()

    def train(self, sess):
        print ('Start training...')
        fout = open(self.out_file, 'w')
        fout.write('Embedding vector size is {}\n'.format(self.args.embeddingSize))
        fout.write('Hidden layer units is {}\n'.format(self.args.hiddenSize))
        fout.write('Learning rate is {}\n'.format(self.args.lrate))
        fout.write('Batch size is {}\n'.format(self.args.batchSize))
        fout.write('Dropout rate is {}\n'.format(self.args.dropoutRate))	 
        fout.close()
        bestVal = 0.0
        bestParams = None
        for e in range(self.args.epochs):
            trainBatches = self.textData.gen_Batches('train')
            y, yp = [], []
            totalTrainLoss = 0.0
            bt = datetime.datetime.now()

            for nextBatch in tqdm(trainBatches):
                self.globalStep += 1

                ops, feed_dict = self.model.step(nextBatch)
                summary, _, loss, predicts = sess.run(ops, feed_dict)
                if np.isnan(loss):
                    print ('Loss of batch {} is Nan'.format(self.globalStep))
                    return
                totalTrainLoss += loss

                y.extend(nextBatch.y.tolist())
                yp.extend(predicts.tolist())
                # self.train_writer.add_summary(summary, self.globalStep)

            et = datetime.datetime.now()

            trainLoss = totalTrainLoss / len(trainBatches)
            trainAcc = accuracy_score(y, yp)
            valAcc = self.test(sess, tag='dev')

            if bestVal <= valAcc:
                bestVal = valAcc
                saveto = os.path.join(self.args.best_dir, 'selector.model')
                saver = tf.train.Saver()
                saver.save(sess, saveto)
                self.save_params()

            print ('epoch = {}/{}, time = {}'.format(e+1, self.args.epochs, et-bt))
            print ('trainLoss = {}, trainAcc = {}, valAcc = {}'.format(trainLoss, trainAcc, valAcc))
            fout = open(self.out_file, 'a')
            fout.write('epoch = {}/{}, time = {}, trainLoss = {}, trainAcc = {}, valAcc = {}\n'.format(e+1, self.args.epochs, et-bt,
                                                                                                     trainLoss, trainAcc, valAcc))
            fout.close()
            val_file = os.path.join(self.out_dir, str(e+1))
            with open(val_file, 'w') as f:
                for key, value in self.textData.val_predict.items():
                    f.write('id: {}\n'.format(key))
                    f.write('Q: {}\n'.format(value[0]))
                    f.write('T: {}\n'.format(value[1]))
                    f.write('P: {}\n'.format(value[2]))
                    f.write('------------------------\n')

    def test(self, sess, tag):
        batches = self.textData.gen_Batches(tag)
        y, yp = [], []
        for nextBatch in batches:
            ops, feed_dict = self.model.step(nextBatch, True)
            predicts = sess.run(ops, feed_dict)
            y.extend(nextBatch.y.tolist())
            yp.extend(predicts.tolist())
            self.textData.write_Result(nextBatch.id, predicts, nextBatch.y)
        return accuracy_score(y, yp)

    def forward(self, sess):
        batches = self.textData.gen_Batches('dev')
        y, yp = [], []
        for nextBatch in batches:
            ops, feed_dict = self.model.step(nextBatch, True)
            predicts = sess.run(ops, feed_dict)
            y.extend(nextBatch.y)
            yp.extend(predicts)
            self.textData.write_Result(nextBatch.id, predicts, nextBatch.y)
        totalAcc = accuracy_score(y, yp)
        print ('Test accuracy is {}'.format(totalAcc))

    def save_params(self):
        print ('Save the best params')
        saveto = os.path.join(self.args.best_dir, 'best')
        pkl.dump(self.textData.word2idx, open('{}_word2idx.pkl'.format(saveto), 'wb'))
        pkl.dump(self.textData.idx2word, open('{}_idx2word.pkl'.format(saveto), 'wb'))
        print ('Save params done')
