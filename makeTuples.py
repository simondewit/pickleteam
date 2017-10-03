import pickle
import sys

def printProgress(iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
	"""
	Call in a loop to create terminal progress bar
	@params:
			iteration   - Required  : current iteration (Int)
			total       - Required  : total iterations (Int)
			prefix      - Optional  : prefix string (Str)
			suffix      - Optional  : suffix string (Str)
	"""
	filledLength    = int(round(barLength * iteration / float(total)))
	percents        = round(100.00 * (iteration / float(total)), decimals)
	bar             = '#' * filledLength + '-' * (barLength - filledLength)
	numbers         = '(' + str(iteration) + '/' + str(total) + ')'
	sys.stdout.write('%s [%s] %s%s %s %s\r' % (prefix, bar, percents, '%', suffix, numbers)),
	sys.stdout.flush()
	if iteration == total:
		print("\n")

def main():
	tweetIDS = list(open('train_semeval2018task2/crawler/data/tweet_by_ID_30_9_2017__06_51_29.txt.ids'))
	tweetLabels = list(open('train_semeval2018task2/crawler/data/tweet_by_ID_30_9_2017__06_51_29.txt.labels'))
	tweetText = list(open('train_semeval2018task2/crawler/data/tweet_by_ID_30_9_2017__06_51_29.txt.text'))
	tweetsAsTuplesFile = open('tweetsAsTuplesFile2.pickle','wb')


	listWithTuples = []
	for i in tweetIDS:
		index = tweetIDS.index(i)
		listWithTuples.append((tweetLabels[index].strip(),tweetText[index].strip()))
		printProgress(index, len(tweetIDS), prefix = 'Progress:', suffix = 'Complete', barLength = 50)
	
	pickle.dump(listWithTuples,tweetsAsTuplesFile)


if __name__ == '__main__':
	main()