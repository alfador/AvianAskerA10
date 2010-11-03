import random
import AvianAsker_A10


nspecies = 200
nattributes = 288


#Example of a bad asker
#Replace this part by your own function
def myAvianAsker(QAs):
	#First five questions are on attributes 
        if len(QAs) < 5:
                Q=random.randint(0,nattributes-1)
	#All others guess the species
	else:
                Q=random.randint(nattributes,nattributes+nspecies)
	return Q

#Open random bird
infile = open("specie_names.txt","r")
train = infile.readlines()
spec_dict={}
for lines in train:
        entry=lines.split()
        spec_dict[entry[0]]= [entry[1],entry[2],entry[3]]
rndbrd = random.randint(1,len(train))

#Open train data
infile = open("dataset.txt","r")
id_dict={}
attr_dict={}
prev_entry = ""
for lines in infile.readlines():
        entry = lines.split()
        if entry[0] != prev_entry:
                attr_dict = {}
        attr_dict[entry[1]]=entry[2]
        id_dict[entry[0]]=attr_dict
        prev_entry = entry[0]

#Open attribute data
infile = open("attributes.txt","r")
Ques_dict = {}
for lines in infile.readlines():
        entry = lines.split()
        attr = entry[1].split("::")
        name = attr[0].split('_')
        Ques_dict[entry[0]] = " ".join(name) + " of " + attr[1]

# Do a query over all birds
total = 0
max_questions = 0

for bird in range(1, len(train)):
    print "**************************"
    print "Testing bird", bird

    QAs = []
    while True:
        Q = AvianAsker_A10.myAvianAsker(QAs)
        if Q-nattributes+1 == bird:
            # Correct guess
            break
        elif Q >= nattributes + nspecies:
            # Invalid question
            raise Exception("Illegal Question")
        elif Q >= nattributes and Q != bird:
            # Incorrect bird guess 
            pass
            A = '0' #incorrect guess
        else:
            # Some attribute question
            A = id_dict[spec_dict[str(bird)][0]][str(Q)]

        QAs.append([Q, A])

    # Add up the number of questions, plus the last question to guess
    # the right bird
    asked = len(QAs) + 1
    total += asked

    if asked > max_questions:
        max_questions = asked 

print "The average is ", float(total) / (len(train) - 1)
print "The max is ", max_questions
