import numpy as np
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.regularizers import l1
from keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Input, Reshape, MaxPooling2D, BatchNormalization, Dropout, Concatenate, Softmax, LeakyReLU

class NomadsBeta:
    def __init__(self,
                 num_chans,
                 checkpoint=None,
                 learning_rate=1e-3,
                 decay=.005):

        self.num_chans = num_chans

        self.model= None
        if checkpoint == None:
            self.model = self.build_model()
        else:
            self.model = load_model(checkpoint)

        self.optimizer = Adam(lr=learning_rate,
                              beta_1=0.8,
                              beta_2=0.9,
                              decay=decay)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer)


    def build_model(self):
        input_tensor = Input((16, 16, self.num_chans))
        conv1 = Conv2D(16,
                       (5, 5),
                       padding='same',
                       activation='relu')(input_tensor)

        batchnorm1 = BatchNormalization()(conv1)

        mpool1 = MaxPooling2D(pool_size=(2, 2))(batchnorm1)

        #path 1
        fc1_in = Flatten()(mpool1)
        fc1 = Dense(1024,
                    activation='relu')(fc1_in)
        path1_out = Dropout(.5)(fc1)

        #path 2
        conv2 =  Conv2D(32,
                       (3, 3),
                       padding='same',
                       activation='relu')(mpool1)

        batchnorm2 = BatchNormalization()(conv2)

        mpool2 = MaxPooling2D(pool_size=(2, 2))(batchnorm2)
        path2_out = Flatten()(mpool2)

        #rejoin paths
        fc2_in = Concatenate()([path1_out, path2_out])

        fc2 = Dense(512,
                    activation='relu')(fc2_in)

        fc2_out = Dropout(.5)(fc2)

        fc3 = Dense(2,
                    activation=None)(fc2_out)

        prediction = Softmax()(fc3)
        return Model(input_tensor, prediction)


    def train_on_batch(self, batch_features, batch_labels):
        return self.model.train_on_batch(batch_features, batch_labels)


    def predict_on_batch(self, batch_features):
        return self.model.predict(batch_features)

    def checkpoint(self, filepath):
        self.model.save(filepath)
