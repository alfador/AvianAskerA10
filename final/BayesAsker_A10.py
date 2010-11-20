'''
Represents an asker for the milestone.  Uses a search tree with alpha-beta
pruning.  This file should be placed in the same directory as
dataset.txt and specie_names.txt
'''

student_image_filename = 'student/student_image.txt'
dataset_filename = 'student/student_dataset.txt'

nspecies = 200
nattributes = 288

class BayesAsker_A10():
    '''Asker using a simplistic Bayes net.'''


    def __init__(self):
        '''
        Initializer.  Loads necessary data from file.
        '''
        species_file = open(student_image_filename, 'r')
        # Lines in this file are of the form:
        # id(197-6032) num(001-200).path/to/image
        # E.G.:
        # 197 129.Song_Sparrow/Song_Sparrow_0010_635571988.jpg
        # Use this file to establish a mapping from id to num-1
        # NOTE: Mapping species numbers down by 1 here so that they start with 0
        id_to_num = {}
        for line in species_file:
            line = line.split()
            id = int(line[0])
            # For some reason, there's a period after the number.  Fortunately,
            # the number is guaranteed to be 3 digits.
            num = int(line[1][0:3]) - 1 # Map to [0, 199]
            id_to_num[id] = num
        species_file.close()

        # Determine a probability distribution
        # Note that the index of id N in this list is N - 1
        # First make a dictionary mapping (species, attribute) to 
        # [[c00, c01, c02], [c10, c11, c12]]
        # Where cij is the count of answer i with certainty j
        # i = 0: attribute not present
        # i = 1: attribute present
        # j = 0: certain
        # j = 1: probable
        # j = 2: guess
        attribute_counts = {}
        for i in range(nspecies):
            for j in range(nattributes):
                attribute_counts[(i, j)] = [[0, 0, 0], [0, 0, 0]]

        # Set the counts
        dataset_file = open(dataset_filename, 'r')
        # Lines in this file are of the form:
        # id(197-6032) attributeId(0-287) present(0-1) certainty(0-2)
        # user(0-1049?)
        for line in dataset_file:
            line = line.split()
            id        = int(line[0])
            attribute = int(line[1])
            present   = int(line[2])
            certainty = int(line[3])
            user      = int(line[4])
            attribute_counts[(id_to_num[id], attribute)][present][certainty] \
                += 1
        dataset_file.close()

        # Construct a probability distribution
        # NOTE: CHANGE CLASSNAME TO WHATEVER PROBABILITY DISTRIBUTION IS
        # BEING USED.
        self.probability = probability(attribute_counts)





# Probability distribution
class probability:
    '''Represents a probability distribution that supports the following
    operations:
        -p(q_1, ..., q_k), the probability of the answers to questions
         Q_1, ..., Q_k being q_1, ..., q_k

        -p(s | q_1, ..., q_k), the probability of species s conditioned on the
         answers q_1, ..., q_k.

    This class is to be treated as an interface, with its methods merely
    indicating the name of the method to implement.
    '''

    def __init__(self, counts):
        '''Initializes this probability distribution, given a dictionary of
        counts.  The keys of counts take the form (species, attribute), and
        the values are [[c00, c01, c02], [c10, c11, c12]], where cij is the
        count of answer i with certainty j.  species and attribute are
        0-indexed.
        '''
        pass


    def prob(self, questions):
        '''Returns the probability p(q_1, ..., q_k) of the given questions.
        questions is a list of (question, answer, confidence).
        '''
        pass

    def cond_prob(self, questions, species):
        '''Returns the probability p(q_1, ..., q_k | s) of the given questions.
        questions is a list of (question, answer, confidence), and species is
        an integer in (0, 199) giving the species of the bird.
        '''
        pass







if __name__ == '__main__':
    asker = BayesAsker_A10()
