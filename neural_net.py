from math import exp
import argparse
import random

THRESHOLD = 0.5     # Define threshold value for outputs

# Define Neuron base class
class Neuron(object):
    def __init__(self, weights):
        self.weights = weights
        self.output = 0
        self.delta = 0

    def transfer(self, inputs):
        activation = self.weights[-1]
        weight_length = len(self.weights)
        for i in range(weight_length-1):
            activation += self.weights[i] * inputs[i]
        self.output = self.calculate_output(activation)
        return self.output

    def transfer_derivative(self):
        return self.output * (1 - self.output)

    def calculate_output(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def __str__(self):
        string = " ".join('%.3f' % weight for weight in self.weights)
        string = string.rstrip() + "\n"
        return string


# Define Output_Neuron class which inherits from Neuron
class Output_Neuron(Neuron):
    def __init__(self, color, weights):
        super().__init__(weights)
        self.color = color
        self.answer = 0
        self.positives = 0
        self.negatives = 0
        self.total_positives = 0
        self.total_negatives = 0

    # Calculate statistics for output neuron
    def display_stats(self):
        if self.total_positives == 0:
            correct = 0
            false_neg = 0
        else:
            correct = 100 * self.positives / self.total_positives
            false_neg = 100 * (1 - self.positives / self.total_positives)

        if self.total_negatives == 0:
            false_pos = 0
        else:
            false_pos = 100 * self.negatives / self.total_negatives

        string = "Neuron: %s\nFired correctly %.3f\nFalse positives: %.3f\nFalse negatives: %.3f\n" \
            % (self.color, correct, false_pos, false_neg)

        print(string)


# Reads weights from file (different n_hidden values require new file)
def read_file(n_inputs, n_hidden, n_outputs):
    output_weights = []
    hidden_weights = []
    shift = 0
    try:
        with open('weights.txt', 'r') as _in:
            lines = _in.readlines()
            lines_length = len(lines)
            for i in range(lines_length):
                weights = []
                line = lines[i].rstrip()
                values = line.split(' ')
                if not shift and len(values) == n_hidden+1:
                    for value in values:
                        weights.append(float(value))
                    output_weights.append(weights)
                elif len(values) == n_inputs+1:
                    for value in values:
                        weights.append(float(value))
                    hidden_weights.append(weights)
                if len(output_weights) == n_outputs:
                    shift = 1
    except IOError:
        print("Unable to access requested file.")
    return (hidden_weights, output_weights)


# Write weights from each neuron into file
def save_file(network):
    try:
        with open('weights.txt', 'w') as _out:
            network_length = len(network)
            output_layer = network[-1]
            _out.write("Output Layer:\n")
            for neuron in output_layer:
                _out.write(str(neuron))
            for i in range(network_length-1):
                layer = network[i]
                _out.write("Layer %d:\n" % (i+1))
                for neuron in layer:
                    _out.write(str(neuron))
    except IOError:
        print("Unable to open requested file.")
        return 1
    return 0


# Display statistics
def display_stats(total, positives, multiples, zeroes):
    string = "Correctly classified: %.3f\nCaused multiple neurons to fire: %.3f\nNo neurons fired: %.3f\nTotal: %d\n" % \
                (100 * positives / total, 100 * multiples / total, 100 * zeroes / total, total)

    print(string)


# Initialize neural network with parameters and whether to randomly generate weights or not
def initialize_network(n_inputs, n_hidden, colors, rand):
    hidden_weights = []
    output_weights = []
    color_length = len(colors)
    if rand:            # Randomly initialize weights
        for i in range(n_hidden):
            temp = []
            for j in range(n_inputs+1):
                temp.append(random.random())
            hidden_weights.append(temp)

        for i in range(color_length):
            temp = []
            for j in range(n_hidden+1):
                temp.append(random.random())
            output_weights.append(temp)
    else:
        (hidden_weights, output_weights) = read_file(n_inputs, n_hidden, color_length)

    network = []
    hidden_layer = [Neuron(hidden_weights[i]) for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [Output_Neuron(colors[i], output_weights[i]) for i in range(color_length)]
    network.append(output_layer)
    return network


# Propagate input forward through network
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            new_inputs.append(neuron.transfer(inputs))
        inputs = new_inputs     # Take outputs of previous layer and use them as inputs for the next
    return inputs


# Propagate error back starting from output layer
def backward_propagate_error(network, expected):
    network_length = len(network)
    for i in reversed(range(network_length)):
        layer = network[i]
        layer_length = len(layer)
        errors = []
        if i != network_length-1:   # Calculate error differently for hidden layers
            for j in range(layer_length):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron.weights[j] * neuron.delta)
                errors.append(error)
        else:
            for j in range(layer_length):
                neuron = layer[j]
                errors.append(expected[j] - neuron.output)
        for j in range(layer_length):
            neuron = layer[j]
            neuron.delta = errors[j] * neuron.transfer_derivative()


# Update weights for each neuron in the network with parameters
def update_weights(network, row, alpha):
    network_length = len(network)
    for i in range(network_length):
        inputs = row
        if i != 0:      # For layers that take the outputs from previous layers
            inputs = [neuron.output for neuron in network[i-1]]
        for neuron in network[i]:
            input_length = len(inputs)
            for j in range(input_length):
                neuron.weights[j] += alpha * neuron.delta * inputs[j]
            neuron.weights[-1] += alpha * neuron.delta


# Adjust necessary statistics of output neurons for visual feedback
def increment_stats(network, expected):
    output_layer = network[-1]
    expected_length = len(expected)
    for i in range(expected_length):
        output = 0
        neuron = output_layer[i]
        if neuron.output >= THRESHOLD:
            output = 1
        if expected[i]:
            neuron.positives += output
            neuron.total_positives += 1
        else:
            neuron.negatives += output
            neuron.total_negatives += 1
        neuron.answer = output


# Runs inputs on the network given parameters
def train_network(network, lines, alpha, decay, epochs, colors):
    total = 0
    positives = 0
    multiples = 0
    zeroes = 0

    for epoch in range(epochs):
        for row in lines:
            inputs = row.rstrip().split(' ')
            if len(inputs) == 4:
                for i in range(3):
                    inputs[i] = float(inputs[i]) / 255  # Scale inputs down to decimals
                answers = []
                outputs = forward_propagate(network, inputs[:-1])
                # Create list of expected outputs
                expected = [1 if inputs[-1] == color else 0 for color in colors]
                backward_propagate_error(network, expected)
                update_weights(network, inputs[:-1], alpha)
                alpha *= decay                      # Apply alpha decay
                increment_stats(network, expected)

                # Calculate statistics for overall output
                for neuron in network[-1]:
                    answers.append(neuron.answer) 
                sum_answers = sum(answers)
                if sum_answers == 0:
                    zeroes += 1
                elif sum_answers > 1:
                    multiples += 1
                else:
                    index = expected.index(1)
                    if answers[index] == 1:
                        positives += 1
                total += 1

    for neuron in network[-1]:
        neuron.display_stats()
    display_stats(total, positives, multiples, zeroes)


# Same as train_network except without calculating errors and adjusting weights
def test(network, lines, colors):
    total = 0
    positives = 0
    multiples = 0
    zeroes = 0

    for row in lines:
        inputs = row.rstrip().split(' ')
        if len(inputs) == 4:
            for i in range(3):
                inputs[i] = float(inputs[i]) / 255
            answers = []
            outputs = forward_propagate(network, inputs[:-1])
            expected = [1 if inputs[-1] == color else 0 for color in colors]
            increment_stats(network, expected)

            for neuron in network[-1]:
                answers.append(neuron.answer) 
            sum_answers = sum(answers)
            if sum_answers == 0:
                zeroes += 1
            elif sum_answers > 1:
                multiples += 1
            else:
                index = expected.index(1)
                if answers[index] == 1:
                    positives += 1
            total += 1

    for neuron in network[-1]:
        neuron.display_stats()
    display_stats(total, positives, multiples, zeroes)


# Main function
def main():
    random.seed()
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', action='store', type=str,
                        required=True, help='Input file name (Required)')
    parser.add_argument('-a', '--alpha', action='store', type=float,
                        help='Training rate', default=0.1)
    parser.add_argument('-e', '--epochs', action='store', type=int,
                        help='Number of epochs', default=1)
    parser.add_argument('-r', '--random', action='store_true', default=False,
                        help='Randomly initialize weights')
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help='Set test mode')
    parser.add_argument('-s', '--shuffle', action='store_true', default=False,
                        help='Shuffle inputs around')
    parser.add_argument('-d', '--decay', action='store', type=float,
                        help='Alpha decay rate', default=1)

    args = parser.parse_args()
    try:
        check_args(args)
    except ValueError as e:
        print(e)
        return 1

    # Initialize color order
    colors = ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange', 'Brown', 'Pink', 'Gray']
    network = initialize_network(3, 9, colors, args.random)

    try:
        with open(args.file, 'r') as _in:
            lines = _in.readlines()
            if not args.test:
                if args.shuffle:        # Shuffle inputs if needed
                    random.shuffle(lines)
                train_network(network, lines, args.alpha, args.decay, args.epochs, colors)
                save_file(network)
            else:
                test(network, lines, colors)
    except IOError:
        print("Unable to access requested file.")
        return 2

    return 0


# Check arguments for invalid values
def check_args(args):
    if (args.alpha < 0 or args.alpha > 1):
        raise ValueError("Error: Alpha can only be between [0,1]")
    if (args.alpha <= 0 or args.alpha > 1):
        raise ValueError("Error: Alpha decay rate can only be between (0,1]")


if __name__ == "__main__":
    main()