import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, matthews_corrcoef
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import label_binarize
from numpy import interp
import tensorflow as tf


def ensemble_metrics_caculation(modelpath, x_test, y_test, label_names, rocPath, cmPath):
    tf.compat.v1.disable_v2_behavior()
    best_m = load_model(modelpath)
    acc_, loss_ = best_m.evaluate(x_test, y_test)
    print(f'best model evaluation: {acc_}, {loss_}')
    y_pred_0 = best_m.predict(x_test)
    y_pred = np.argmax(y_pred_0, axis=1)
    print(classification_report(np.argmax(y_test, axis=1), y_pred, digits=4))
    

    # Binarize ypreds with shape (n_samples, n_classes) 
    ypreds = label_binarize(y_pred, classes=[0,1,2,3,4,5,6,7,8,9])
    auc_roc = roc_auc_score(y_test, y_pred_0, multi_class='ovo')
    print(f'the auc score is {auc_roc}')
    mcc = matthews_corrcoef(np.argmax(y_test, axis=1), y_pred)
    print(f'mcc is {mcc}')
    # visualization

    # draw roc curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        #fpr[i], tpr[i], _ = roc_curve(y_test[:, i], ypreds[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_0[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), ypreds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(10):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 10

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig = plt.figure()
    fig.set_size_inches(8, 8, forward=True)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=2)

    colors = plt.get_cmap('Paired')(np.linspace(0, 1, 10))
    for i, color in zip(range(10), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of {0} (area = {1:0.4f})'
                       ''.format(label_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel('False Positive Rate', fontdict={'size': 16})
    plt.ylabel('True Positive Rate', fontdict={'size': 16})
    plt.title('ROC performance of each class', fontdict={'size':20})
    plt.legend(loc="lower right")
    plt.savefig(rocPath)
    plt.show()
    plt.close()

    # plot confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontdict={'size': 16})
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, size=10)
        plt.yticks(tick_marks, classes, size=10)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontdict={'size': 16})
        plt.xlabel('Predicted label', fontdict={'size': 16})

    # Plot normalized confusion matrix
    fig = plt.figure()
    fig.set_size_inches(11, 11)
    #fig.align_labels()

    # fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    plot_confusion_matrix(cm, classes=np.asarray(label_names), normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(cmPath, bbox_inches = 'tight')
    plt.tight_layout()
    plt.show()
    plt.close()
   '
