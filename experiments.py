from lp1_model import *


MODELS = {
    'MLSTM-FCN':    generate_model(),
    'MALSTM-FCN':   generate_model_2(),
    'LSTM-FCN':     generate_model_3(),
    'ALSTM-FCN':    generate_model_4()
}

if __name__ == "__main__":

    with open('results.txt', 'w+') as output:
        for model_name in MODELS.keys():
            model = MODELS[model_name]
            for i in range(3):
                train_model(model, DATASET_INDEX, dataset_prefix='lp1_', epochs=1000, batch_size=128)
                accuracy, loss = evaluate_model(model, DATASET_INDEX, dataset_prefix='lp1_', batch_size=128)
                output.write('%s;%s;%d;%f;%f\n' % ('lp1', model_name, i, accuracy, loss))

