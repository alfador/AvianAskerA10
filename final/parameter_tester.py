import random
import AvianAsker_A10 
import cPickle
import sys
import os
import math

nspecies = 200
nattributes = 288

#Open attribute data
infile = open("./student/attributes.txt","r")
Ques_dict = {}
for lines in infile.readlines():
        entry = lines.split()
        attr = entry[1].split("::")
        name = attr[0].split('_')
        Ques_dict[entry[0]] = " ".join(name) + " of " + attr[1]

image = {}
spec_dict={}
#Open image dirs
image_dir = open("./student/student_image.txt","r")
for i in image_dir.readlines():
        entry = i.split()
        image[int(entry[0])] = entry[1]
        specy = entry[1].split(".")
        name = specy[1].split("/")
        if int(specy[0]) not in spec_dict.keys():
                spec_dict[int(specy[0])] = name[0]

#Open dataset
### NOTE: THIS LINE TAKES ABOUT 3.5 MINUTES ON MY COMPUTER
print 'Loading dataset...'
dataset = open("./student/student_cPickle.txt","r")
data = cPickle.load(dataset)
print 'Done loading dataset.'
dataset.close()

# Set output file
out = sys.stdout
if '--hide' in sys.argv:
    out = file(os.devnull, 'w')

def std_dev(nums):
    '''Calculates the standard deviation of a list of numbers.'''
    mean = float(sum(nums)) / len(nums)
    diffs_squared = [(num - mean) ** 2 for num in nums]
    return math.sqrt(1. / (len(nums) - 1) * sum(diffs_squared))


### LIST PARAMETERS HERE
parameter_sets = []
for diffuse in [.0001, .1, 1.]:
    for diffusions in [5]:
        for species_bias in [.33, 1., 1.66]:
            parameter_sets.append((diffuse, diffusions, species_bias))

### File to write to
output_file = open('parameter_results.txt', 'a')
n = 100
output_file.write('Number of birds per parameter set: %d\n' % n)

# TODO: Threading
for parameter in parameter_sets:
    diffuse      = parameter[0]
    diffusions   = parameter[1]
    species_bias = parameter[2]
    AvianAsker_A10.diffuse_multiplier = diffuse
    AvianAsker_A10.num_diffusions = diffusions
    AvianAsker_A10.species_guess_bias = species_bias
    output_file.write(str(parameter) + ": ")
    print 'On parameter set: ', parameter

    #Begin new game
    Sum = 0
    num_questions = []
    print 'Initializing asker...'
    AA = AvianAsker_A10.AvianAsker_A10()##()
    for i in range(n):
            image_id = random.choice([k for k in image.keys()])
            rndbrd = int((image[image_id].split("."))[0])
            AA.init()
            myAvianAsker = AA.myAvianAsker
            QAs = []
            while True:
                    Q = myAvianAsker(image[image_id], QAs)
                    if Q-nattributes+1 == rndbrd:
                            print >> out, ("Your score is "+str(len(QAs)+1))
                            Sum = Sum + len(QAs) + 1
                            num_questions.append(len(QAs))
                            break
                    elif Q >= nattributes + nspecies:
                            print >> out, ("The question is out of range")
                            continue
                    elif Q >= nattributes and Q != rndbrd:
                            print >> out, ("Incorrect guess")
                            A = '0' #incorrect guess
                    else:
                            if Q in data[image_id].keys():
                                    A = random.choice(data[image_id][Q])
                            else:
                                    A = '2'
    
                    QAs.append([Q, A])
    
            print "Num is " + str(i+1) + ", Sum is "+str(Sum)+", Score now is "+str(float(Sum)/(i+1))
    score = float(Sum) / n
    output_file.write(str(score) + ', ' + str(std_dev(num_questions)) + '\n')
    output_file.flush()

output_file.close()
