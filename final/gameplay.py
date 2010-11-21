import random
from BayesAsker_A10 import *## import *
import cPickle

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
dataset = open("./student/student_cPickle.txt","r")
data = cPickle.load(dataset)
dataset.close()

#Begin new game
Sum = 0
n = 100
AA = BayesAsker_A10()##()
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
                        print("Is it "+spec_dict[Q-nattributes+1]+"?")
                        print("You have guessed correctly, the bird is "+spec_dict[rndbrd]+"\n")
                        print("Your score is "+str(len(QAs)+1))
                        Sum = Sum + len(QAs) + 1
                        break
                elif Q >= nattributes + nspecies:
                        print("The question is out of range")
                        continue
                elif Q >= nattributes and Q != rndbrd:
                        print("Is it "+spec_dict[Q-nattributes+1]+"?")
                        print("Sorry, you are wrong!\n")
                        A = '0' #incorrect guess
                else:                
                        print("It "+ Ques_dict[str(Q)] +"?")
                        if Q in data[image_id].keys():
                                A = random.choice(data[image_id][Q])
                        else:
                                A = '2'
                        print(str(Q)+" "+str(A))
                        if A == [1,0]:
                                print("Yes! Probably.\n")
                        elif A == [1,1]:
                                print("Yes! Definitely.\n")
                        elif A == [1,2]:
                                print("Yes! I guess.\n")
                        elif A == [0,0]:
                                print("No! Probably.\n")
                        elif A == [0,1]:
                                print("No! Definitely.\n")
                        elif A == [0,2]:
                                print("No! I guess.\n")
                        else:
                                print("I don't know!\n")

                QAs.append([Q, A])
        print("Num is " + str(i+1) + ", Sum is "+str(Sum)+", Score now is "+str(Sum/(i+1)))
print("Your final score is "+str(Sum/n))

