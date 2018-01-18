import pickle
import sklearn


def readFile(file):
	return pickle.load(open(file, 'rb'))


def createLists(documents):
	tweets = []
	categories = []
	for category, tweet in documents:
		categories.append(category)
		# if "…" in tweet:
		# 	tweet = tweet.replace("…","...")
		tweets.append(tweet)

	return tweets, categories

def main():

	chosenFeatrue = "max" #"sum"

	svmfile = readFile('........pickle')
	kerasfile = readFile('........pickle')

	SVMemoji = [line[0] for line in open(svmfile,"r+").read().split("\n")]
	SVMprob  = [line[1] for line in open(svmfile,"r+").read().split("\n")]

	KERASemoji = [line[0] for line in open(kerasfile,"r+").read().split("\n")]
	KERASprob  = [line[1] for line in open(kerasfile,"r+").read().split("\n")]

	guess_categories = []


	with open("english.output.txt") as finalFile:
		for idx, emoji in enumerate(SVMemoji):
			if SVMemoji[idx] == KERASemoji[idx]:
				finalFile.write(SVMemoji[idx]+"\n")
				guess_categories.append(SVMemoji[idx])

			else:
				if chosenFeatrue == "max":
					if max(SVMprob[idx]) >= max(KERASprob[idx]):
						finalFile.write(SVMemoji[idx]+"\n")
						guess_categories.append(SVMemoji[idx])
					else:
						finalFile.write(KERASemoji[idx]+"\n")
						guess_categories.append(KERASemoji[idx])

				else:
					sumList = []
					for i, elem in enumerate(SVMprob[idx]):
						total = SVMprob[idx][i] + KERASprob[idx][i]
						sumList.append(total)

					emojiNR = sumList.index(max(sumList))
					finalFile.write(emojiNR+"\n")
					guess_categories.append(emojiNR[idx])



	test_documents = readFile('es-trial.pickle')
	test_tweets, test_categories = createLists(test_documents)

	#average scores
	precisionScoreAverage = sklearn.metrics.precision_score(test_categories,guess_categories, average="macro")
	print("\n\nprecision sklearn:", round(precisionScoreAverage,3))

	recallScoreAverage = sklearn.metrics.recall_score(test_categories,guess_categories, average="macro")
	print("recall sklearn:", round(recallScoreAverage,3))

	f1ScoreAverage = sklearn.metrics.f1_score(test_categories,guess_categories, average="macro")
	print("fscore sklearn:", round(f1ScoreAverage,3),"\n")


	#scores per class
	labels = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19"]
	print("Class \t Precision \t Recall \t F-score")
	for label in labels:
		precisionScore = sklearn.metrics.precision_score(test_categories,guess_categories, average="macro", labels=label)
		recallScore = sklearn.metrics.recall_score(test_categories,guess_categories, average="macro", labels=label)
		f1Score = sklearn.metrics.f1_score(test_categories,guess_categories, average="macro", labels=label)

		print(label, "\t", round(precisionScore,3), "\t\t", round(recallScore,3), "\t\t", round(f1Score,3), "\t")


	print()








if __name__ == '__main__':
	main()