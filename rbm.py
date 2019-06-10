#math para calculos
import math
#Tensorflow para os modelos de machine learn
import tensorflow as tf
#Numpy para calculos matematicos
import numpy as np
#Image para manipulacao de imagens
from PIL import Image
#importar image
from utils import tile_raster_images


# definindo a classe maquina de boltzman restrita
class RBM(object):

    def __init__(self, input_size, output_size):
        # definindo hiperparametros
        self._input_size = input_size
        self._output_size = output_size
        self.epochs = 5  # numero de interacoes do treino
        self.learning_rate = 1.0  # passo usado no gradiente descendente
        self.batchsize = 100  # quantia de dados aserem usados no treino por sub-interacao

        # Iniciando os pesos e bias como matrizes de zeros
        self.w = np.zeros([input_size, output_size], np.float32)  # inicializando os pesos com zero
        self.hb = np.zeros([output_size], np.float32)  # inicializando as bias ocultas com zero
        self.vb = np.zeros([input_size], np.float32)  # inicializando as bias visiveis com zero

    # ajustando o resultado da camada visivel(aplicada a esta os pesos) e as bias, em uma curva sigmoid
    def prob_h_given_v(self, visible, w, hb):
        # Sigmoid
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    # ajustando o resultado da camada oculta(aplicada a esta os pesos) e as bias, em uma curva sigmoid
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    # gerando a probabilidade por amostra
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    # metodo de treino para o modelo
    def train(self, X):
        # criando espaco exclusivo para os parametros
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])

        anterior_w = np.zeros([self._input_size, self._output_size],
                         np.float32)  # Creates and initializes the weights with 0
        anterior_hb = np.zeros([self._output_size], np.float32)  # Creates and initializes the hidden biases with 0
        anterior_vb = np.zeros([self._input_size], np.float32)  # Creates and initializes the visible biases with 0

        atual_w = np.zeros([self._input_size, self._output_size], np.float32)
        atual_hb = np.zeros([self._output_size], np.float32)
        atual_vb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder("float", [None, self._input_size])

        # inicializando com as probabilidades por amostra
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        # criando gradientes
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        # atualizando a taxa(pontuacao) de aprendizado por camada
        update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vb = _vb + self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self.learning_rate * tf.reduce_mean(h0 - h1, 0)

        # calcular erro
        err = tf.reduce_mean(tf.square(v0 - v1))

        # loop do treino
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # para cada epoch
            for epoch in range(self.epochs):
                # para cada passo step/batch
                for start, end in zip(range(0, len(X), self.batchsize), range(self.batchsize, len(X), self.batchsize)):
                    batch = X[start:end]
                    # atualizando pontuacao
                    atual_w = sess.run(update_w, feed_dict={v0: batch, _w: anterior_w, _hb: anterior_hb, _vb: anterior_vb})
                    atual_hb = sess.run(update_hb, feed_dict={v0: batch, _w: anterior_w, _hb: anterior_hb, _vb: anterior_vb})
                    atual_vb = sess.run(update_vb, feed_dict={v0: batch, _w: anterior_w, _hb: anterior_hb, _vb: anterior_vb})
                    anterior_w = atual_w
                    anterior_hb = atual_hb
                    anterior_vb = atual_vb
                error = sess.run(err, feed_dict={v0: X, _w: atual_w, _vb: atual_vb, _hb: atual_hb})
                print ('Epoch: %d' % epoch, 'reconstruction error: %f' % error)
            self.w = anterior_w
            self.hb = anterior_hb
            self.vb = anterior_vb

    # saida necessaria para a DBN
    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)