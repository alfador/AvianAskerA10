'''
Represents an asker for the milestone.  Uses a search tree with alpha-beta
pruning.  This file should be placed in the same directory as
dataset.txt and specie_names.txt
'''

species_filename = 'specie_names.txt'
dataset_filename = 'dataset.txt'

nspecies = 200
nattributes = 288

class AvianAskerA10():
    '''
    Represents an asker for the milestone.  Uses a search tree with alpha-beta
    pruning.
    '''


    def __init__(self):
        '''
        Initializer.  Loads necessary data from file.
        '''
        species_file = open(species_filename, 'r')
        # Lines in this file are of the form:
        # num(1-200) id(13-6015) name image_name
        # Use this file to establish a mapping from id to num
        id_to_num = {}
        for line in species_file:
            line = line.split()
            id_to_num[int(line[1])] = int(line[0])
        species_file.close()

        # Get the attributes of each bird, setting:
        #   0 - Attribute is false
        #   1 - Attribute is true
        #   2 - Attribute is unknown 
        # Note that the index of id N in this list is N - 1
        # Dimensions (nattributes + 1) x nspecies
        self.attributes = [[0] * nspecies for _ in range(nattributes)]

        dataset_file = open(dataset_filename, 'r')
        # Lines in this file are of the form:
        # id(13-6015) attributeId(0-287) value(0-1)
        # Missing id/attributeId pairs are unknown
        last_id = -1
        for line in dataset_file:
            line = line.split()
            id        = int(line[0])
            attribute = int(line[1])
            value     = int(line[2])
            self.attributes[attribute][id_to_num[id] - 1] = value
        dataset_file.close()


    def ask_question(self, questions):
        '''
        Given a list of question and answer pairs each of the form [Q, A],
        returns the number of a question to ask.
        A question Q in the range 0-(nattributes-1) corresponds to asking if
        the bird has attribute Q.
        If Q is in the range (nattributes)-(nspecies+nattributes), then
        the question correspond to asking about bird number
        Q - nattributes + 1.
        '''
        # List of which bird species are viable.  True is viable, False is
        # not viable
        viable = [True] * nspecies
        for [q, a] in questions:
            viable = self._update_viable(viable, q, a)

        # Determine the number of current viable birds.  If there are only 4
        # or fewer viable birds, just start guessing.
        # Note: Might need to change this if only allowed one guess.
        current_viable = viable.count(True)
        #print 'current viable: %d' % current_viable
        if current_viable <= 4:
            bird_number = viable.index(True) + 1
            #print 'Guessing bird number: %d' % bird_number
            # Convert to range (288-487)
            return bird_number + nattributes - 1


        # Now we need to decide on a good question to guess.
        # For now, just ask the question that minimizes the maximum number
        # viable species after the question, accounting for all possible
        # answers.
        best_size = nspecies
        best_question = -1
        # Traverse questions in decreasing numerical order in order to give
        # priority to asking about the bird itself.
        for q in range(nspecies + nattributes - 1, -1, -1):
            max_size = 0
            for a in [0, 1, 2]:
                max_size = max(max_size,
                    self._update_viable(viable, q, a).count(True))
            if max_size < best_size:
                best_size = max_size
                best_question = q
        #print 'best question: %d, max remaining viable: %d' % \
            (best_question, best_size)

        return best_question


    def _update_viable(self, viable, question, answer):
        '''
        Returns an updated list of viable species, given that the answer to
        question 'question' is 'answer'.
        The list is comprised of 0s and 1s, where 1 means viable, 0 means not
        viable.

        A question Q in the range 0-(nattributes-1) corresponds to asking if
        the bird has attribute Q.
        If Q is in the range (nattributes)-(nspecies+nattributes), then
        the question correspond to asking about bird number
        Q - nattributes + 1.

        An answer A is 0 for false (does not have attribute), 1 for true, and
        2 for unsure.
        '''
        # Determine whether the question was about a specific species or
        # an attribute
        if question >= nattributes:
            species_question = question - nattributes + 1
            if answer == 0:
                new_viable = viable[:]
                new_viable[species_question - 1] = False
                return new_viable
            elif answer == 1:
                # This code shouldn't execute, since if we were right we
                # shouldn't have to ask another question.
                new_viable = [False] * nspecies
                new_viable[species_question - 1] = True
                return new_viable
            else:
                # Not sure which bird it is.  This shouldn't happen, otherwise
                # it would be impossible to test anything.  In this case, just
                # don't eliminate anything
                return viable[:]

        # Question corresponded to an attribute
        attribute_num = question
        attribute = self.attributes[attribute_num]

        # Update viable list.  The birds that are now viable should include
        # only the birds that were already viable which also have a matching
        # attribute.
        new_viable_func = lambda i: viable[i] and attribute[i] == answer
        new_viable = map(new_viable_func, range(nspecies))
        return new_viable


    def _print_ids(self, viable, val):
        '''
        Prints the ids(1-200) of the entries in the given viable list that
        match 'val'.  Used for debugging purposes.
        '''
        print 'Ids that are %s:' % val
        for i in range(len(viable)):
            if viable[i] == val:
                print "%d," % (i + 1),
        print


def myAvianAsker(questions):
    '''
    Queries the asker for a question to ask.
    Input:
        questions - List of [question, answer]
    Returns:
        A question to ask.
    '''
    # Convert inputs to integers because we might be passed in strings (wtf)
    questions = [[int(a), int(b)] for [a, b] in questions]

    #print 'got questions: %s' % questions
    question = asker.ask_question(questions)
    #print 'asking question %d' % question
    return question


# Make this global so the initialization code doesn't have to run at every
# question.
asker = AvianAskerA10()

if __name__ == '__main__':
    asker = AvianAskerA10()
    asker.ask_question([])
