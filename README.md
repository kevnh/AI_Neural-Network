# AI_Neural-Network
Used https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/ as reference.\
Color recognition neural network made for Artificial Intelligence course.

Typing:\
python neural_net.py -h\
will give a list of additional flags.

Defaults are:\
alpha = 0.1\
epochs = 1\
random = False	(Determines whether to read weights from file "weights.txt" or randomly initialize weights)\
test = False	(Simply runs the inputs in the neural net without any of the training steps and ignores many irrevelvant parameters)\
shuffle = False	(Determines whether to shuffle the inputs or read the inputs in order from the input file\
decay = 1

Constants:\
THRESHOLD = 0.5

IMPORTANT NOTES:\
Input file is required.\
weights.txt is updated when training, there is no way\
to change the saved text file name without changing it\
in the .py file.

Network:\
3 - User Inputs\
9 - Layer 1 Nodes\
9 - Output Nodes
