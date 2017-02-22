from numpy import exp, array, random ,dot

class NeuralNetwork():
    def __init__(self):
        # seed the random number
        random.seed(1)

        #single neuron with 3 input and 1 output connections
        #assign random weights to the matrix , with values ranging from -1 to 1
        self.synaptic_weights = 2 * random.random((3,1)) - 1

    # the activation function which sigmoid
    def __sigmoid(self, x):
        return 1/(1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1-x)

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):

			output = self.predict(training_set_inputs)

			error = training_set_outputs - output

			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
			self.synaptic_weights += adjustment
	def printt(self):
		print "hello"

    def predict(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights ))




if __name__ == "__main__":

    #initialize a single neuron neural network
    myneural_network = NeuralNetwork()
    # myneural_network.printt()

    print 'Random starting synaptic weights:'
    print myneural_network.synaptic_weights

    #the training set . We have 4 examples , each consisting of 3 input values
    # and 1 output values.
    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    #train the neural_network using a training set and interating this 10000 to adjust the weight each time

    myneural_network.train(training_set_inputs,training_set_outputs, 10000)

    print 'New synaptic weights after training: '
    print myneural_network.synaptic_weights

    #test the neural network
    print 'Considering new situation [1,0,0] -> ?:'
    print myneural_network.predict(array([1,0,0]))
