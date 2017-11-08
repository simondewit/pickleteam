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
	tweetText = list(open('trialdata/es_trial.text'))
	tweetLabels = list(open('trialdata/es_trial.labels'))
	tweetsAsTuplesFile = open('es-trial.pickle', 'wb')



	listWithTuples = []
	print(tweetLabels)
	for idx, i in enumerate(tweetLabels):
		listWithTuples.append((i.strip(),tweetText[idx].strip()))
		# printProgress(index, len(tweetLabels), prefix = 'Progress:', suffix = 'Complete', barLength = 50)
	
	pickle.dump(listWithTuples,tweetsAsTuplesFile)


if __name__ == '__main__':
	main()