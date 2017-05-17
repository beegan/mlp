from random import uniform
from math import tanh

class MultiLayerPerceptron(object):
    def __init__(self, inputs, hidden, outputs, initial_weight_range):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = []
        self.layers.append(Layer(inputs))
        self.layers.append(Layer(hidden))
        self.layers.append(Layer(outputs))
        self.randomise(initial_weight_range)
        self.output = []

    def randomise(self, weight_range):
        layer_num = 0
        for layer in self.layers:
            try:
                parent = self.layers[layer_num+1]
                for unit in layer.units:
                    for link in parent.units:
                        unit.weight[link] = uniform(-weight_range, weight_range)
                        unit.weight_change[link] = 0
            except IndexError:
                pass
            layer_num += 1

    def forward(self, i):
        self.set_input(i)
        layer_num = 0
        for layer in self.layers:
            try:
                upper_layer = self.layers[layer_num + 1]
                for upper_unit in upper_layer.units:
                    output = 0.0
                    for current_unit in layer.units:
                        output += current_unit.activation * current_unit.weight[upper_unit]
                    upper_unit.activation = self.squash(output)
            except IndexError:
                pass #at output layer so no next layer
            layer_num += 1 #increment for next upper_payer reference
        self.output = []
        for output_unit in self.layers[-1].units:
            self.output = self.output + [output_unit.activation]
        return self.output


    def backwards(self, target):
        output_layer_num = len(self.layers) - 1
        for layer_num in range(output_layer_num, -1, -1): #counts backwards through layers
            if layer_num is output_layer_num:
                component = 0
                output_layer = self.layers[layer_num]
                for output_unit in output_layer.units:
                    error = target[component] - output_unit.activation
                    output_unit.delta = error * self.deriv_squash(output_unit.activation)
                    if layer_num - 1 >= 0:
                        lower_layer = self.layers[layer_num - 1] #gets outermost hidden layer & calcs weight changes
                        for lower_unit in lower_layer.units:
                            lower_unit.weight_change[output_unit] += output_unit.delta * lower_unit.activation
                    component += 1
            else: #for all non-output layers
                for current_unit in self.layers[layer_num].units:
                    error = 0.0
                    try:
                        higher_layer = self.layers[layer_num + 1]
                        for higher_unit in higher_layer.units:
                            error += higher_unit.delta * current_unit.weight[higher_unit]
                    except IndexError:
                        print("No layer found above layer %d" % layer_num)  # safeguard for assuming higher layer exists
                        pass
                    current_unit.delta = error * self.deriv_squash(current_unit.activation)
                    if layer_num - 1 >= 0:
                        lower_layer = self.layers[layer_num - 1] #gets next layer lower (input for 3-layer network)
                        for lower_unit in lower_layer.units:
                            lower_unit.weight_change[current_unit] += current_unit.delta * lower_unit.activation

        squared_error = 0.0
        i = 0
        for output_unit in self.layers[-1].units:
            squared_error += 0.5 * (target[i] - output_unit.activation)**2
        return squared_error


    def update_weights(self, learning_rate):
        for layer in self.layers:
            for unit in layer.units:
                for higher_unit in unit.weight_change:
                    unit.weight[higher_unit] += learning_rate * unit.weight_change[higher_unit]
                    unit.weight_change[higher_unit] = 0
        #print(self)

    def set_input(self, inp):
        input_layer = self.layers[0]
        x = 0
        for unit in input_layer.units:
            unit.activation = inp[x]
            x += 1

    def deriv_squash(self, activation):
        return 1 - activation ** 2

    def squash(self, x):
        return tanh(x)

    def __str__(self):
        str = ""
        x = 0
        for layer in self.layers:
            str = "%s\nLayer[%s]" % (str, x)
            y = 0
            for unit in layer.units:
                str = "%s\n\t Unit[%s]: %s" % (str, y, unit.activation)
                z = 0
                for weight in unit.weight:
                    str = "%s\n\t\t Weight[%d]: %s" % (str, z, unit.weight[weight])
                    z += 1
                y += 1
            x += 1
        return str

class Unit(object):
    def __init__(self):
        self.weight = {}
        self.weight_change = {}
        self.delta = None
        self.activation = None

class Layer(object):
    def __init__(self, numberOfUnits):
        self.units = []
        for n in range(0, numberOfUnits):
            self.units.append(Unit())
