import tensorflow as tf


class Bitparm(tf.keras.Model):
    '''
    save params
    '''
    def __init__(self, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = tf.Variable(tf.random.normal([1,1], mean=0.0, stddev=0.01),trainable=True, name = 'h')
        self.b = tf.Variable(tf.random.normal([1,1], mean=0.0, stddev=0.01),trainable=True, name = 'b')
        if not final:
            self.a = tf.Variable(tf.random.normal([1,1], mean=0.0, stddev=0.01),trainable=True , name = 'a')
        else:
            self.a = None

    def call(self, x):
        if self.final:
            return tf.keras.activations.sigmoid(x * tf.keras.activations.softplus(self.h) + self.b)
        else:
            x = x * tf.keras.activations.softplus(self.h) + self.b
            return x + tf.keras.activations.tanh(x) * tf.keras.activations.tanh(self.a)
        
class bit_estimator(tf.keras.Model):
    '''
    Estimate bit
    '''
    def __init__(self):
        super(bit_estimator, self).__init__()
        self.f1 = Bitparm()
        self.f2 = Bitparm()
        self.f3 = Bitparm()
        self.f4 = Bitparm(True)
        
    def call(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)
