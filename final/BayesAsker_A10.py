'''
Represents an asker for the milestone.  Uses a search tree with alpha-beta
pruning.  This file should be placed in the same directory as
dataset.txt and specie_names.txt
'''

import math

student_image_filename = 'student/student_image.txt'
dataset_filename = 'student/student_dataset.txt'

nspecies = 200
nattributes = 288

class BayesAsker_A10():
    '''Asker using a simplistic Bayes net.'''
    # TODO: This doesn't really use a Bayes net, so rename it.


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

        # Determine priors
        # TODO: Should this really be uniform?
        self.priors = [1. / nspecies] * nspecies


    def init(self):
        '''Prepares this BayesAsker for a new bird.'''
        # self.distribution = self.priors[:]
        # For now, all the necessary code is recalculated each time.
        pass


    # TODO: Use the image
    def myAvianAsker(self, image, questions):
        '''Given an image and list of questions, returns a new question to ask.
        Arguments:
            image: The path to the image.
            questions: List of [question, [present, confidence]].  question
                ranges between 0 and 487, present is 0 (not present) or 1
                (present), and confidence is 0 (certain), 1 (probable), or
                2 (guess).

        Return value:
            An integer Q between 0 and 487, inclusive.  If Q is between 0 and
            287, the question is asking about an attribute, and if Q is between
            288 and 487, the question is asking whether it is bird Q - 287.
        '''
        # Since this method is called after every question, we only need to
        # update our distribution for the last question asked.

        # For now, just update the distribution to take into account all
        # questions
        distribution = self.priors[:]

        # First, parse the questions, removing questions that correspond to
        # asking what the bird is.
        # This will be a list of question, answer pairs
        usable_questions = []
        bad_questions = [] # Questions which returned an answer of 'don't know'
        for [question, answer] in questions:
            # The answer is either a string '0' for incorrect guess, '2' for
            # don't know, or a list [present, confidence]
            if type(answer) is str:
                answer = int(answer)
                if answer == 0:
                    # Incorrect guess
                    species = question - nattributes # [0, 199]
                    distribution[species] = 0. # Definitely not that bird
                elif answer == 2:
                    # Can't get anything out of this question, maybe.
                    # Note that this case doesn't imply uncertainty, just a lack
                    # of data.  We shouldn't ask this question again.
                    bad_questions.append(question)

            # Otherwise, the answer takes the form [present, confidence].
            # Just add the question to the questions that have been asked.
            usable_questions.append([question, answer])

        # Since we may have changed the probability distribution, re-normalize
        p_sum = sum(distribution)
        distribution = [p / p_sum for p in distribution]

        # Determine the current probability distribution over birds.
        # Bayes' Rule: p(s | q^k) = p(q^k | s) * p(s) / p(q^k)
        prob_questions = self.probability.prob(usable_questions)
        posterior_distribution = [0] * len(distribution)
        for i, prob in enumerate(distribution):
            posterior_distribution[i] = self.probability.cond_prob(
                usable_questions, i) * prob / prob_questions

        # If any of the probabilities are above some threshold, guess that bird
        max_prob = max(distribution)
        if max_prob > .25:
            # TODO: Optimize this value
            best_species = distribution.index(max_prob)
            guess = best_species + nattributes
            return guess

        # Pick the question that maximizes mutual information.  That is,
        # argmax_Y I(S ; Y | q^k)
        # I(S ; Y | q^k) = H(S | q^k) - H(S | q^k, Y)
        # argmin_Y H(S | q^k, Y)
        # H(S | q^k, Y) = sum_y p(y) * H(S | q^k, y)
        # H(S | q^k, y) = sum_s p(s | q^k, y) lg (1 / p(s | q^k, y))
        # argmin_Y sum_y p(y) * sum_s p(s | q^k, y) lg(1 / p(s | q^k, y))
        # argmax_Y sum_y p(y) * sum_s p(s | q^k, y) lg(p(s | q^k, y))
        # Bayes' Rule: p(s | q^k, y) = p(q^k, y | s) * p(s) / p(q^k, y)

        max_neg_entropy = -nspecies # Negative entropy bounded below by -lg(|S|)
        best_question = -1
        for Y in xrange(nattributes): # argmax Y
            # Don't ask questions we got 'don't know' back from
            if Y in bad_questions:
                continue
            neg_entropy = 0
            for y in [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]: # sum_y
                prob_y = self.probability.prob([[Y, y]]) # p(y)
                usable_questions.append([Y, y])
                prob_qk_y = self.probability.prob(usable_questions) # p(q^k, y)
                for s in xrange(nspecies): # sum_s
                    # TODO: For a significant speedup, take these out of
                    # function calls and hard-code into the for loop.
                    prob_qk_y_s = self.probability.cond_prob(usable_questions,
                        s) # p(q^k, y | s)
                    # distribution takes into account wrong guesses, which
                    # are not part of the usable questions.
                    prob_s = distribution[s] # p(S)
                    prob_s_qk_y = prob_qk_y_s * prob_s / prob_qk_y
                    # Should divide by log(2), but this isn't necessary for
                    # argmax
                    neg_entropy += prob_s_qk_y * math.log(prob_s_qk_y)
                usable_questions.pop()
            if neg_entropy > max_neg_entropy:
                max_neg_entropy = neg_entropy
                best_question = Y

        # Ask the best question
        return best_question



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
        questions is a list of [question, [answer, confidence]].
        '''
        pass

    def cond_prob(self, questions, species):
        '''Returns the probability p(q_1, ..., q_k | s) of the given questions.
        questions is a list of [question, [answer, confidence]], and species is
        an integer in (0, 199) giving the species of the bird.
        '''
        pass


class dummy_probability(probability):
    '''Returns probabilities that are valid but meaningless.  Only used for
    testing.'''

    def __init__(self, counts):
        '''Initializes this probability distribution, given a dictionary of
        counts.  The keys of counts take the form (species, attribute), and
        the values are [[c00, c01, c02], [c10, c11, c12]], where cij is the
        count of answer i with certainty j.  species and attribute are
        0-indexed.
        '''
        # Each possible answer will be equally probable.  Since there are 6
        # possible [present, confidence] pairs, 
        self.base_prob = 1. / 6
        pass


    def prob(self, questions):
        '''Returns the probability p(q_1, ..., q_k) of the given questions.
        questions is a list of [question, [answer, confidence]].
        '''
        # All answers are uniformly probable
        return self.base_prob ** len(questions)

    def cond_prob(self, questions, species):
        '''Returns the probability p(q_1, ..., q_k | s) of the given questions.
        questions is a list of [question, [answer, confidence]], and species is
        an integer in (0, 199) giving the species of the bird.
        '''
        # Independent of species
        return self.base_prob ** len(questions)



if __name__ == '__main__':
    asker = BayesAsker_A10()
    asker.init()
    asker.probability = dummy_probability(None)
    import time
    start_time = time.time()
    print asker.myAvianAsker('image', [])
    print "time taken to ask question: ", time.time() - start_time
    import cProfile
    asker.init()
    cProfile.run('asker.myAvianAsker("image", [])')