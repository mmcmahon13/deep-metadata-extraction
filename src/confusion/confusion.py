import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools

def make_confusion_matrix(y_true, y_pred, labels_str_id_map, labels_id_str_map):
    print(type(y_true[0]))
    print(np.array(labels_str_id_map.keys()))
    cm = confusion_matrix(y_true, y_pred, np.array(labels_str_id_map.keys()))
    plot_confusion_matrix(cm, labels_str_id_map.keys(), labels_id_str_map, normalize=True)

def plot_confusion_matrix(cm, classes, labels_id_str_map,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #[labels_id_str_map[c] for c in classes]
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def compute_f1_score(ytrue, ypred, tag_set):
    # this is direct from the Meta example script Shankar sent
    tag_level_metrics = dict()

    # get the types without the BIO
    ytrue = np.array([y.split('-')[1] if y != 'O' else y for y in ytrue])
    ypred = np.array([y.split('-')[1] if y != 'O' else y for y in ypred])

    for tag in tag_set:
        ids = np.where(ytrue == tag)[0]
        if len(ids) == 0: continue
        yt = np.zeros(len(ytrue))
        yp = np.zeros(len(ytrue))
        yt[ids] = 1
        yp[np.where(ypred == tag)] = 1

        tp = np.dot(yp, yt)
        fn = len(ids) - tp
        fp = sum(yp[np.setdiff1d(np.arange(len(ytrue)), ids)])

        if tp == 0:
            tag_level_metrics[tag] = (0, 0, 0)
        else:
            p = tp * 1. / (tp + fp)
            r = tp * 1. / (tp + fn)
            f1 = 2. * p * r / (p + r)
            tag_level_metrics[tag] = (p, r, f1)

    return tag_level_metrics

def main():
    with open('label.txt', 'r') as f:
        labels_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        labels_id_str_map = {i: s for s, i in labels_str_id_map.items()}
    y_pred = np.load('test_preds.npy')
    y_true = np.load('test_labels.npy')
    flat_preds = np.concatenate([p.flatten() for p in y_pred])
    flat_labels = np.concatenate([l.flatten() for l in y_true])

    accuracy = sum(flat_preds == flat_labels) * 1. / len(flat_labels)
    print(accuracy)

    make_confusion_matrix(flat_labels, flat_preds, labels_str_id_map, labels_id_str_map)

    tag_set = set([l.split('-')[-1] for l in labels_str_id_map.keys()])
    print(tag_set)
    tag_level_metrics = compute_f1_score(y_true, y_pred, tag_set)
    for tag in tag_level_metrics:
        print('Precision, Recall, F1 for ' + str(tag) + ': ' + str(tag_level_metrics[tag][0]) + ', ' + str(
            tag_level_metrics[tag][1]) + ', ' + str(tag_level_metrics[tag][2]))

if __name__ == '__main__':
    main()