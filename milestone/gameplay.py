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

#Begin new game
QAs = []
while True:
	Q = AvianAsker_A10.ask(QAs)
	if Q-nattributes+1 == rndbrd:
                print("Is it "+" ".join(spec_dict[str(Q-nattributes+1)][1].split('_'))+"?")
                print("You have guessed correctly, the bird is "+" ".join(spec_dict[str(rndbrd)][1].split('_'))+"\n")
                break
        elif Q >= nattributes + nspecies:
                print("The question is out of range")
                continue
        elif Q >= nattributes and Q != rndbrd:
                print("Is it "+" ".join(spec_dict[str(Q-nattributes+1)][1].split('_'))+"?")
                print("Sorry, you are wrong!\n")
                A = '0' #incorrect guess
        else:                
                print("It "+ Ques_dict[str(Q)] +"?")
                A = id_dict[spec_dict[str(rndbrd)][0]][str(Q)];                       
                if A == '1':
                        print("Yes!\n")
                else:
                        print("No!\n")

	QAs.append([Q, A])
