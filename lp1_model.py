from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM

DATASET_INDEX = 41

MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

TRAINABLE = True


def generate_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out, name='MLSTM_FCN')
    model.summary()

    # add load model code here to fine-tune

    return model


def generate_model_2():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    # stride = 10

    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    #ip1 = K.reshape(ip,shape=(MAX_TIMESTEPS,MAX_NB_VARIABLES))
    #x = Permute((2, 1))(ip)
    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out, name='MALSTM_FCN')
    model.summary()

    # add load model code here to fine-tune

    return model


def generate_model_3():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out, name='LSTM_FCN')
    model.summary()

    # add load model code here to fine-tune

    return model


def generate_model_4():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    # stride = 3
    #
    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out, name='ALSTM_FCN')
    model.summary()

    # add load model code here to fine-tune

    return model


def squeeze_excite_block(input):
    '''
    Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


# test james
from keras import layers
from extra.rbflayer import RBFLayer
from utils.generic_utils import load_dataset_at, calculate_dataset_metrics
import numpy as np
import scipy


def generate_model_5():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    # features
    x = concatenate([x, y])

    model1 = build_rbf(ip, x)   # Model(ip, out1)
    model2 = build_rbf(ip, x)   # Model(ip, out2)
    model3 = build_rbf(ip, x)   #

    ensemble = ensemble_models([
        model1,
        model2,
        model3
    ], ip)
    ensemble.summary()

    #model = build_mlp(ip, x)
    #model.summary()

    # add load model code here to fine-tune

    #return model
    #from keras.utils.vis_utils import plot_model
    #plot_model(ensemble, to_file='model.png')
    return ensemble


def build_mlp(input_problem, output_features):
    l1 = Dense(64,
               kernel_initializer='random_uniform',
               bias_initializer='uniform',
               activation='relu')(output_features)
    out = Dense(NB_CLASS, activation='softmax')(l1)

    model = Model(input_problem, out)

    return model


def build_rbf(input_problem, output_features):
    l1 = RBFLayer(NB_CLASS*3, 0.5)(output_features)
    out = Dense(NB_CLASS, activation='softmax')(l1)

    model = Model(input_problem, out)

    return model


def ensemble_models(models, model_input):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models]
    # averaging outputs
    yAvg = layers.average(yModels)

    out = Dense(NB_CLASS, activation='softmax')(yAvg)

    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=out, name='ensemble')

    return modelEns


def exec_ensemble(models):
    # train models
    for m in models:
        train_model(m, DATASET_INDEX, dataset_prefix='lp1_', epochs=1000, batch_size=128)

    # Predict labels with models
    _, _, X_test, y_test, is_timeseries = load_dataset_at(DATASET_INDEX)
    labels = []
    for m in models:
        predicts = np.argmax(m.predict(X_test), axis=1)
        labels.append(predicts)

    # Ensemble with voting
    labels = np.array(labels)
    labels = np.transpose(labels, (1, 0))
    labels_result = scipy.stats.mode(labels, axis=1)[0]
    labels_result = np.squeeze(labels_result)

    # cal accuracy
    match_count = 0
    for i in range(len(labels_result)):
        p = labels_result[i]
        e = y_test[i][0]
        if p == e:
            match_count += 1

    accuracy = (match_count * 100) / len(labels_result)
    print('Accuracy: ', accuracy)

    return accuracy

if __name__ == "__main__":
    # model = generate_model_5()
    # train_model(model, DATASET_INDEX, dataset_prefix='lp1_', epochs=1000, batch_size=128)
    # evaluate_model(model, DATASET_INDEX, dataset_prefix='lp1_', batch_size=128)

    model1 = generate_model()
    model2 = generate_model_2()
    # model3 = generate_model_3()
    model4 = generate_model_4()

    models = [
        model1,
        model2,
        # model3,
        model4
    ]
    accs = []
    for i in range(4):
        acc = exec_ensemble(models)
        accs.append(acc)

    print(accs)







