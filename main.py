#!/usr/bin/python

from MultiLayerPerceptron import MultiLayerPerceptron
import getopt
import sys
from datetime import datetime
from random import uniform
from math import sin
from string import ascii_uppercase

class Assignment(object):
    def __init__(self, args):
        super(Assignment, self).__init__()
        self.args = args
        self.XOR = False
        self.VECTOR = False
        self.WRITING = True
        self.maxEpoch = 10000
        self.main()

    def main(self):
        self.get_args()
        if not (self.XOR or self.VECTOR or self.WRITING):
            self.XOR = True #run in XOR mode if no mode specified

        if self.XOR:
            MLP = MultiLayerPerceptron(2, 2, 1, 0.2)
            data =  [
                        [[0, 0], [0]],
                        [[0, 1], [1]],
                        [[1, 0], [1]],
                        [[1, 1], [0]]
                    ]
            file_name = "outputs/xor%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S")
            f = open(file_name, 'w')
            for i in range(self.maxEpoch):
                error = 0
                for j in range(len(data)):
                    MLP.forward(data[j][0])
                    error += MLP.backwards(data[j][1])
                f.write("Epoch, %d, Error, %f\n" % (i, error))
                MLP.update_weights(0.09)
            f.close()

            print("XOR test")
            print("Final Error after %d epochs: %f" % (self.maxEpoch, error))
            print("Input (0,0), expected 0, actual %s" % MLP.forward((0,0)))
            print("Input (0,1), expected 1, actual %s" % MLP.forward((0,1)))
            print("Input (1,0), expected 1, actual %s" % MLP.forward((1,0)))
            print("Input (1,1), expected 0, actual %s" % MLP.forward((1,1)))


        if self.VECTOR:
            data = self.generate_vectors()
            training = data[:40]
            test = data[40:]

            MLP = MultiLayerPerceptron(4, 5, 1, 0.5)
            file_name = "outputs/sin_vector%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S")
            f = open(file_name, 'w')
            for i in range(self.maxEpoch):
                error = 0
                for j in range(len(training)):
                    MLP.forward(training[j][0])
                    error += MLP.backwards(training[j][1])
                f.write("Epoch, %d, Error, %f\n" % (i, error))
                MLP.update_weights(0.05)
            f.close()

            print("Sine Vector test")
            print("Final Error after %d epochs: %f" % (self.maxEpoch, error))
            for item in test:
                print("Input, %s, Expected, %s, Actual, %s" % (item[0], item[1], MLP.forward(item[0])))

        if self.WRITING:
            data = self.read_writing()
            training = data[:4000]
            test = data[4000:5000]
            self.maxEpoch = 1000

            MLP = MultiLayerPerceptron(16, 10, 26, 0.5)
            file_name = "outputs/letters%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S")
            f = open(file_name, 'w')
            for i in range(self.maxEpoch):
                error = 0
                for j in range(len(training)):
                    MLP.forward(training[j][0])
                    error += MLP.backwards(training[j][1])
                f.write("Epoch, %d, Error, %f\n" % (i, error))
                print("Epoch, %d, Error, %f" % (i, error))
                MLP.update_weights(0.3)
            f.close()

            print("Writing test")
            file_name = "outputs/letters-tests%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S")
            f = open(file_name, 'w')
            f.write("Input, Expected Letter, Letter Guess, Expected Vector, Actual Vector\n")
            print("Final Error after %d epochs: %f" % (self.maxEpoch, error))
            for item in test:
                out_vector = MLP.forward(item[0])
                out_string = "%s,%s,%s,%s,%s,%s" % (item[0], item[2], self.vector_to_letter(out_vector), item[1], out_vector)
                f.write(out_string)
                print(out_string)

    def generate_vectors(self):
        pattern_list = []
        for x in range(50):
            pattern = []
            vector = []
            for i in range(4):
                vector.append(uniform(-1,1))
            pattern.append(vector)
            result = []
            result.append(sin(sum(vector)))
            pattern.append(result)
            pattern_list.append(pattern)
        return pattern_list

    def read_writing(self):
        f = open("inputs/letter-recognition.data", 'r')
        pattern_list = []
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            elements = line.split(',')
            pattern = []
            pattern.append([int(x) for x in elements[1:]])
            output = [0.0] * 26
            letter_num = ascii_uppercase.index(elements[0])
            output[letter_num] = 1.0
            pattern.append(output)
            pattern.append(elements[0])
            pattern_list.append(pattern)
        return pattern_list

    def vector_to_letter(self, vect):
        try:
            letter_num = vect.index(1.0)
        except ValueError:
            return vect
        return ascii_uppercase[letter_num]

    def get_args(self):
        try:
            opts, args = getopt.getopt(self.args, "f:xvw")
        except getopt.GetoptError:
            print("Invalid option given")
            self.printhelp()
            sys.exit(2)
        for o, a in opts:
            if o == "-x":
                self.XOR = True
            elif o == "-v":
                self.VECTOR = True
            elif o == "-w":
                self.WRITING = True # for handwriting dataset but not done
            else:
                assert False, "Invalid option"

if __name__ == '__main__':
    lr = Assignment(sys.argv[1:]) #removes script name from args and passes them down