import numpy as np
import sklearn
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from cairocffi import *

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
		
	#print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def main(test_categories,predicted_categories):
	y_true = test_categories
	y_pred = predicted_categories
	labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']

	matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels)
	#print(matrix)

	matrix_normalized = np.transpose(np.transpose(matrix) / matrix.astype(np.float).sum(axis=1))
	#print(matrix_normalized.round(2)) #horizontal = predicted; vertical = true label

	for item in labels:
		if item == labels[0]:
			print("{:>8s}".format(item), end="")
		else:
			print("{:>8s}".format(item), end="")

	for i,line in enumerate(matrix_normalized):
		print()
		print("{:<2s}".format(labels[i]), end="")
		for item in line:
			print("{:8.2f}".format(item), end="")


	print()
	# #Only available in Linux:
	# plt.figure()
	# plot_confusion_matrix(matrix, classes=labels, title='Confusion matrix')
	# plt.show()


if __name__ == '__main__':
	main()
