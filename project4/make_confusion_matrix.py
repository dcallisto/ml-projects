import sys
import numpy as np

if __name__ == '__main__':
    sub_file = sys.argv[1]
    target_file = sys.argv[2]
    submission = np.loadtxt(sub_file, delimiter=',')
    targets = np.loadtxt(target_file, delimiter=',')

    confusion_matrix = np.zeros((10, 10))

    for i in range(0, targets.shape[0]):
    	prediction = np.argmax(submission[i])
    	answer = np.argmax(targets[i])
    	confusion_matrix[answer][prediction] += 1
    np.savetxt("confusion_matrix.csv", confusion_matrix, delimiter = ',')