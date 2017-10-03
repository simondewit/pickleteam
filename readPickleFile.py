import pickle
import sys

def main():

	with open('tweetsAsTuplesFile2.pickle','rb') as f:
		tweetsAsTuplesFile = pickle.load(f)

		for i in tweetsAsTuplesFile:
			print(i)



if __name__ == '__main__':
	main()