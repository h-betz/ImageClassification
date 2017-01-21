from math import *
import sys
import time
import datetime
import knearestneighbor

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70

def column(A, j):
   return [row[j] for row in A]

def formatTestData(datafilename, labelfilename, width, height):
	labels = []

	with open(labelfilename) as lab:
		for line in lab:
			labels.append(line)
	data_str = []
	image = []
	with open(datafilename) as f:
		line_num = 1
		count = 0
		for line in f:
    		#For each line do the following
			if line_num % height == 0:
				data_str = ''.join(data_str)
				data_str += ','
				num = ''.join(labels[count])
				data_str += ' '
				data_str += num
				image.append(data_str)
				count += 1
				data_str = []
			else:
				#Append to string
				if line > 1:
					data_str.append(' ')
				for c in line:
					if c == ' ':
						data_str.append('0')
					else:
						data_str.append('1')
					data_str.append(' ')
			data_str = data_str[:-1]
			line_num += 1

	data = [line.strip().split(',') for line in image]
	data = [([int(x) for x in point.split()], int(label)) for (point, label) in data]
	return data

def formatData(datafilename, labelfilename, width, height):
	labels = []

	with open(labelfilename) as lab:
		for line in lab:
			line = line.replace(" ", "")
			labels.append(line)
	data_str = []
	image = []
	with open(datafilename) as f:
		line_num = 1
		count = 0
		for line in f:
    		#For each line do the following
			if line_num % height == 0:
				data_str = ''.join(data_str)
				data_str += ','
				num = ''.join(labels[count])
				data_str += ' '
				data_str += num
				image.append(data_str)
				count += 1
				data_str = []
			else:
				#Append to string
				if line > 1:
					data_str.append(' ')
				for c in line:
					if c == ' ':
						data_str.append('0')
					else:
						data_str.append('1')
					data_str.append(' ')
			line_num += 1		
			data_str = data_str[:-1]
			

	data = [line.strip().split(',') for line in image]
	data = [([int(x) for x in point.split()], int(label)) for (point, label) in data]

	return data


if __name__ == '__main__':
	start_time = time.time()
	if sys.argv[1] == '-f':
		#Load faces data
		limit = int(sys.argv[2])
		k = int(sys.argv[3])
		face_train_data = 'facedata/facedatatrain'
		face_train_label = 'facedata/facedatatrainlabels'
		face_test_data = 'facedata/facedatatest'
		face_test_label = 'facedata/facedatatestlabels'
		training_data = formatData(face_train_data, face_train_label, 60, 70)
		test_data = formatTestData(face_test_data, face_test_label, 60, 70)
		pts, labels = column(training_data, 0), column(training_data, 1)
		testpts, testlabels = column(test_data, 0), column(test_data, 1)
		xpts = pts[:limit]
		xlab = labels[:limit]
		from knearestneighbor import KNearestNeighbor
		knn = KNearestNeighbor()
		knn.train(xpts, xlab, testpts, testlabels, k)
	else:
		#Load digit data
		limit = sys.argv[1]
		k = int(sys.argv[2])
		digit_train_data = 'digitdata/trainingimages'
		digit_train_label = 'digitdata/traininglabels'
		digit_test_data = 'digitdata/testimages'
		digit_test_label = 'digitdata/testlabels'
		training_data = formatData(digit_train_data, digit_train_label, 28, 28)
		test_data = formatTestData(digit_test_data, digit_test_label, 28, 28)
		pts, labels = column(training_data, 0), column(training_data, 1)
		testpts, testlabels = column(test_data, 0), column(test_data, 1)
		limit = int(limit)
		xpts = pts[:limit]
		xlab = labels[:limit]
		from knearestneighbor import KNearestNeighbor
		knn = KNearestNeighbor()
		knn.train(xpts, xlab, testpts, testlabels, k)
	print("--- %s seconds ---" % (time.time() - start_time))
