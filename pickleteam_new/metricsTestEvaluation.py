import numpy as np
import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from createConfusionMatrix import main as mainCCM

def main():

	""" beste resulatten met eerste inzending (SVM volgens mij) """

	predictedTestUS = open("2018_04/13_02_07/english_predicted.output.txt", 'rb').readlines()
	goldTestUS 		= open("test_semeval2018task2_gold/us_test.labels", 'rb').readlines()

	predictedTestES = open("2018_04/13_35_30/spanish_predicted.output.txt", 'rb').readlines()
	goldTestES 		= open("test_semeval2018task2_gold/es_test.labels", 'rb').readlines()



	for language in ["English", "Spanish"]:
		if language == "English":
			gold = [str(int(line.strip())) for line in goldTestUS]
			predicted = [str(int(line.strip())) for line in predictedTestUS]
			labels = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19"]
		

		else:
			gold = [str(int(line.strip())) for line in goldTestES]
			predicted = [str(int(line.strip())) for line in predictedTestES]
			labels = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18"]
		


		print("\n\n--------------",language,"--------------")

		print("Class\tF-score")
		for label in labels:
			label = [label]

			precisionScore 	= sklearn.metrics.precision_score(gold,predicted, average="macro", labels=label)
			recallScore 	= sklearn.metrics.recall_score(gold,predicted, average="macro", labels=label)
			f1Score 		= sklearn.metrics.f1_score(gold,predicted, average="macro", labels=label)

			print("{}\t{:6.3f}".format(label[0],round(f1Score*100,3)))


		f1ScoreAverage 			= sklearn.metrics.f1_score(gold,predicted, average="macro")
		precisionScoreAverage 	= sklearn.metrics.precision_score(gold,predicted, average="macro")
		recallScoreAverage 		= sklearn.metrics.recall_score(gold,predicted, average="macro")
		accuracyAverage 		= sklearn.metrics.accuracy_score(gold,predicted)

		print("\nF1-score:\t", round(f1ScoreAverage*100,3))		
		print("Precision:\t", round(precisionScoreAverage*100,3))		
		print("Recall:\t\t", round(recallScoreAverage*100,3))		
		print("Accuracy:\t", round(accuracyAverage*100,3),"\n")


		mainCCM(gold,predicted,labels)

		# break



if __name__ == '__main__':
	main()