import random
from BayesAsker_A10 import *## import *
import cPickle
import sys
import os
import time

nspecies = 200
nattributes = 288

#TA's asker
#Replace this part by your own function in another file
def myAvianAsker(image,QAs):
        return random.randint(1,487)

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
start_time = time.time()
dataset = open("./student/student_cPickle.txt","r")
data = cPickle.load(dataset)
dataset_loading_time = time.time() - start_time
print 'Done loading dataset.'
dataset.close()

# Set output file
out = sys.stdout
if '--hide' in sys.argv:
    out = file(os.devnull, 'w')

#Begin new game
Sum = 0
n = 100
print 'Initializing asker...'
start_time = time.time()
AA = BayesAsker_A10()##()
asker_initialization_time = time.time() - start_time
print 'Done initializing asker.'
start_time = time.time()
for i in range(n):
        image_id = random.choice([k for k in image.keys()])
        rndbrd = int((image[image_id].split("."))[0])
        print 'DEBUG: BIRD IS: ', rndbrd
        AA.init()
        myAvianAsker = AA.myAvianAsker
        QAs = []
        while True:
                Q = myAvianAsker(image[image_id], QAs)
                if Q-nattributes+1 == rndbrd:
                        print >> out, ("Is it "+spec_dict[Q-nattributes+1]+"?")
                        print >> out, ("You have guessed correctly, the bird is "+spec_dict[rndbrd]+"\n")
                        print >> out, ("Your score is "+str(len(QAs)+1))
                        Sum = Sum + len(QAs) + 1
                        break
                elif Q >= nattributes + nspecies:
                        print >> out, ("The question is out of range")
                        continue
                elif Q >= nattributes and Q != rndbrd:
                        print >> out, ("Is it "+spec_dict[Q-nattributes+1]+"?")
                        print >> out, ("Sorry, you are wrong!\n")
                        A = '0' #incorrect guess
                else:
                        print >> out, ("It "+ Ques_dict[str(Q)] +"?")
                        if Q in data[image_id].keys():
                                A = random.choice(data[image_id][Q])
                        else:
                                A = '2'
                        print(str(Q)+" "+str(A))
                        if A == [1,0]:
                                print >> out,("Yes! Probably.\n")
                        elif A == [1,1]:
                                print >> out,("Yes! Definitely.\n")
                        elif A == [1,2]:
                                print >> out,("Yes! I guess.\n")
                        elif A == [0,0]:
                                print >> out,("No! Probably.\n")
                        elif A == [0,1]:
                                print >> out,("No! Definitely.\n")
                        elif A == [0,2]:
                                print >> out,("No! I guess.\n")
                        else:
                                print >> out,("I don't know!\n")

                QAs.append([Q, A])

                f = open("QA.txt", "w")
                f.write(str(QAs))
                f.close()
        print "Num is " + str(i+1) + ", Sum is "+str(Sum)+", Score now is "+str(Sum/(i+1))
asking_questions_time = time.time() - start_time
print "Your final score is "+str(Sum/n)
print "Time to load dataset: %g" % dataset_loading_time
print "Time to initialize asker: %g" % asker_initialization_time
print "Time taken to ask %d questions: %g" % (Sum, asking_questions_time)
print "Average time per question: %g" % (asking_questions_time / Sum)
print "Average time per bird: %g" % (asking_questions_time / n) 
