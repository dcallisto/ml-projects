import sys
import numpy as np
from log_reg import LogisticRegression




def calculate_overall_accuracy(prob_matrix, correct_labels):

	output_matrix = np.zeros((prob_matrix.shape[0], prob_matrix.shape[1]))
	num_right = 0
	for i in range(0, prob_matrix.shape[0]):
		class_prediction = np.argmax(prob_matrix[i])
		output_matrix[i][class_prediction] = 1
		num_right += np.dot(output_matrix[i], correct_labels[i])
	print("Total accuracy: ")
	print(num_right / correct_labels.shape[0])


def predict(prob_matrix):
	output_matrix = np.zeros((prob_matrix.shape[0], prob_matrix.shape[1]))
	for i in range(0, prob_matrix.shape[0]):
		class_prediction = np.argmax(prob_matrix[i])
		output_matrix[i][class_prediction] = 1
	np.savetxt("submission3.csv", output_matrix, delimiter = ',')





if __name__ == '__main__':
	features_file = sys.argv[1]
	targets_file = sys.argv[2]
	test_file = sys.argv[3]
	test_features = np.loadtxt(features_file, delimiter=',')
	test_targets = np.loadtxt(targets_file, delimiter = ',')
	actual_test_features = np.loadtxt(test_file, delimiter = ',')

	one_targets = test_targets[:, 0]
	two_targets = test_targets[:, 1]
	three_targets = test_targets[:, 2]
	four_targets = test_targets[:, 3]
	five_targets = test_targets[:, 4]
	six_targets = test_targets[:, 5]
	seven_targets = test_targets[:, 6]
	eight_targets = test_targets[:, 7]
	nine_targets = test_targets[:, 8]
	ten_targets = test_targets[:, 9]

	#append features with column of ones for bias term
	test_features = np.append(test_features, np.full((test_features.shape[0], 1), 1), axis=1)
	actual_test_features = np.append(actual_test_features, np.full((actual_test_features.shape[0], 1), 1), axis=1)





	#                        			  Alpha (learning rate)

	#					     .07	    .04  	    .025     	 .01 		.001 

	#125 iterations		|	.6807	|	.6862	|			|	.6706	|	.6498
	#250 iterations		|	.6957	|	.6977	|	.6992	|	.7035	|	.7027
	#500 iterations		|			|	.7543	|	.7562	|	.7605	|
	#600 iterations		|			|	.7282	|			|	.7455	|
	#					|			|			|			|			|
	#set learning rate
	log_reg_one = LogisticRegression("one", .01, 500)
	log_reg_two = LogisticRegression("two", .01, 500)
	log_reg_three = LogisticRegression("three", .01, 500)
	log_reg_four = LogisticRegression("four", .01, 500)
	log_reg_five =  LogisticRegression("five", .01, 500)
	log_reg_six =  LogisticRegression("six", .01, 500)
	log_reg_seven = LogisticRegression("seven", .01, 500)
	log_reg_eight = LogisticRegression("eight", .01, 500)
	log_reg_nine = LogisticRegression("nine", .01, 500)
	log_reg_ten = LogisticRegression("ten", .01, 500)

	log_reg_one.fit(test_features, one_targets)
	log_reg_two.fit(test_features, two_targets)
	log_reg_three.fit(test_features, three_targets)
	log_reg_four.fit(test_features, four_targets)
	log_reg_five.fit(test_features, five_targets)
	log_reg_six.fit(test_features, six_targets)
	log_reg_seven.fit(test_features, seven_targets)
	log_reg_eight.fit(test_features, eight_targets)
	log_reg_nine.fit(test_features, nine_targets)
	log_reg_ten.fit(test_features, ten_targets)

	#predict and calculate accuracy
	#log_reg_one.calculate_accuracy(test_features, one_targets)
	#log_reg_two.calculate_accuracy(test_features, two_targets)
	#log_reg_three.calculate_accuracy(test_features, three_targets)
	#log_reg_four.calculate_accuracy(test_features, four_targets)
	#log_reg_five.calculate_accuracy(test_features, five_targets)
	#log_reg_six.calculate_accuracy(test_features, six_targets)
	#log_reg_seven.calculate_accuracy(test_features, seven_targets)
	#log_reg_eight.calculate_accuracy(test_features, eight_targets)
	#log_reg_nine.calculate_accuracy(test_features, nine_targets)
	#log_reg_ten.calculate_accuracy(test_features, ten_targets)

	prob_matrix = []

	prob_matrix.append(log_reg_one.calculate_prob_for_given_model(actual_test_features))
	prob_matrix.append(log_reg_two.calculate_prob_for_given_model(actual_test_features))
	prob_matrix.append(log_reg_three.calculate_prob_for_given_model(actual_test_features))
	prob_matrix.append(log_reg_four.calculate_prob_for_given_model(actual_test_features))
	prob_matrix.append(log_reg_five.calculate_prob_for_given_model(actual_test_features))
	prob_matrix.append(log_reg_six.calculate_prob_for_given_model(actual_test_features))
	prob_matrix.append(log_reg_seven.calculate_prob_for_given_model(actual_test_features))
	prob_matrix.append(log_reg_eight.calculate_prob_for_given_model(actual_test_features))
	prob_matrix.append(log_reg_nine.calculate_prob_for_given_model(actual_test_features))
	prob_matrix.append(log_reg_ten.calculate_prob_for_given_model(actual_test_features))

	prob_matrix = np.transpose(prob_matrix)

	#calculate_overall_accuracy(prob_matrix, test_targets)
	predict(prob_matrix)








	



	









 