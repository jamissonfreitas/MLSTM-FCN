import lp1_model as lp1
import lp2_model as lp2
import lp3_model as lp3
import lp4_model as lp4
import lp5_model as lp5
from utils.generic_utils import load_dataset_at
import numpy as np
import scipy

EPOCHS = 10
SAMPLES = 5


def exec_ensemble(models, problem):
    print('#Problem: ', problem)
    # train models
    for m in models:
        prefix = problem.__name__.replace('model','')
        problem.train_model(m, dataset_id=problem.DATASET_INDEX,
                            dataset_prefix=prefix, epochs=EPOCHS, batch_size=128)

    # Predict labels with models
    _, _, X_test, y_test, is_timeseries = load_dataset_at(problem.DATASET_INDEX)
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
    # print('Accuracy: ', accuracy)
    return accuracy


def running_ensemble(models, problem):
    accs = []
    for i in range(SAMPLES):
        acc = exec_ensemble(models, problem)
        accs.append(acc)

    return accs


if __name__ == "__main__":

    with open('ensemble_results.txt', 'w+') as output:
        for problem in [lp1, lp2]:

            model1 = problem.generate_model()
            model2 = problem.generate_MALSTM()
            model3 = problem.generate_FCN()

            accs = running_ensemble([model1, model2, model3], problem)

            for acc in accs:
                output.write('%s;%f\n' % (problem.__name__.replace('_model',''), acc))
