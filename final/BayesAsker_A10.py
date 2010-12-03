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

# Parameters
prob_cutoff = .25 # If the max probability is above this, guess a bird.
diffuse_multiplier = .01 # Parameters for diffuse regularization
num_diffusions = 5

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

        # Determine priors
        # TODO: Should this really be uniform?
        self.priors = [1. / nspecies] * nspecies

        # Construct a probability distribution
        # NOTE: CHANGE CLASSNAME TO WHATEVER PROBABILITY DISTRIBUTION IS
        # BEING USED.
        self.probability = independent_probability_diffuse(attribute_counts,
           self.priors)

        # The set of questions already asked
        self.asked_questions = set()

    def init(self):
        '''Prepares this BayesAsker for a new bird.'''
        # Reset the distribution to the original priors
        self.probability.reset_prob()

        # Reset the set of asked questions
        self.asked_questions = set()

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

        # Condition on the last question, if there is a last question to
        # condition on.
        if len(questions) > 0:
            question, answer = questions[-1]

            # Add the question to the set of asked questions
            self.asked_questions.add(question)

            if answer == '0': 
                # Incorrect bird guess, should remove that species
                self.probability.remove_bird(question - nattributes)
            elif answer == '2':
                # A don't know answer, which means no data was present.
                # In this case, there is nothing to condition on
                pass
            else:
                # In this case, an answer to an attribute was obtained
                self.probability.cond_on_question(questions[-1])

        posterior_distribution = self.probability.get_probs()

        dist_sum = sum(posterior_distribution)
        print 'Sum prob: ', dist_sum

        max_prob = max(posterior_distribution)
        print 'Max prob: ', max_prob
        # Calculate current entropy and print it out.
        entropy = 0
        for p in posterior_distribution:
            if p != 0:
                entropy += p * math.log(1. / p)
        print 'Species entropy: ', entropy / math.log(2)

        # If any of the probabilities are above some threshold, guess that bird
        if max_prob > prob_cutoff:
            # TODO: Optimize this value
            best_species = posterior_distribution.index(max_prob)
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
        log = math.log # Minor speedups
        cond_prob = self.probability.cond_prob
        # Ignore attribute 0, which we have almost no data on.
        for Y in xrange(1, nattributes): # argmax Y
            # Don't ask questions that were already asked
            if Y in self.asked_questions:
                continue

            neg_entropy = 0
            for y in [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]: # sum_y
                prob_y = self.probability.prob([[Y, y]]) # p(y)
                # If the probability of getting these answers is 0, we'll be
                # dealing with nonsense distributions.
                # TODO: Does it make more sense to just continue through this
                # for loop or continue through the outer one too?
                if prob_y == 0:
                    continue
                for s in xrange(nspecies): # sum_s
                    # TODO: For a significant speedup, take these out of
                    # function calls and hard-code into the for loop.
                    prob_y_s = cond_prob([[Y, y]], s) # p(y | s)

                    prob_s = posterior_distribution[s] # p(S)
                    prob_s_y = prob_y_s * prob_s / prob_y

                    # Should divide by log(2), but this isn't necessary for
                    # argmax
                    # The limit of what we're adding as prob_s_y -> 0
                    # is 0, so don't add anything in that case.
                    if prob_s_y != 0:
                        # sum_y p(y) * sum_s p(s | y) lg(p(s | y))
                        neg_entropy += prob_y * prob_s_y * log(prob_s_y)

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

    def __init__(self, counts, priors):
        '''Initializes this probability distribution, given a dictionary of
        counts.  The keys of counts take the form (species, attribute), and
        the values are [[c00, c01, c02], [c10, c11, c12]], where cij is the
        count of answer i with certainty j.  species and attribute are
        0-indexed.  priors is a list of prior probabilities on the species.
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

    def reset_prob(self):
        '''
        Reset the distribution on s to the distribution on construction
        '''
        pass

    def cond_on_question(self, question):
        '''
        Modify the distribution on s by conditioning on the answer to a
        particular question
        '''
        pass

    def remove_bird(self, bird):
        '''
        Remove all the probability from an incorrect bird
        '''
        pass 

    def get_probs(self):
        '''
        Returns the current distribution on birds
        '''
        pass


class dummy_probability(probability):
    '''Returns probabilities that are valid but meaningless.  Only used for
    testing.'''

    def __init__(self, counts, priors):
        '''Initializes this probability distribution, given a dictionary of
        counts.  The keys of counts take the form (species, attribute), and
        the values are [[c00, c01, c02], [c10, c11, c12]], where cij is the
        count of answer i with certainty j.  species and attribute are
        0-indexed.  priors is a list of prior probabilities on the species.
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

    def get_probs(self):
        '''
        Returns the current distribution on birds
        '''
        return [1.0 / nspecies for i in range(nspecies)]

class independent_probability(probability):
    '''Represents a probability distribution where the attributes are all
    independent.'''

    # IMPLEMENTATION NOTES:
    # IF THE GIVEN COUNTS DON'T INCLUDE ANY INSTANCES OF A (SPECIES, ATTRIBUTE)
    # PAIR, THOSE PROBABILITIES ARE SET TO 0.


    def __init__(self, counts, priors):
        '''Initializes this probability distribution, given a dictionary of
        counts.  The keys of counts take the form (species, attribute), and
        the values are [[c00, c01, c02], [c10, c11, c12]], where cij is the
        count of answer i with certainty j.  species and attribute are
        0-indexed.  priors is a list of prior probabilities.
        '''
        self.priors = priors

        # Get probabilities p(q | s)
        # Probabilities are indexed by [species][attribute]
        self.species_attributes = [[None] * nattributes for _ in
                                   range(nspecies)]
        for key in counts.keys():
            (species, attribute) = key
            [c0, c1] = counts[key]
            total_count = float(sum(c0) + sum(c1))
            total_count = max(total_count, 1)
            p0 = [c / total_count for c in c0]
            p1 = [c / total_count for c in c1]
            self.species_attributes[species][attribute] = [p0, p1]

        # Get probabilities p(q) = sum_s p(s) p(q | s)
        self.attributes = [None] * nattributes
        for att in xrange(nattributes):
            prob_sum = [[0., 0., 0.], [0., 0., 0.]]
            for species in xrange(nspecies):
                # [[p00, p01, p02], [p10, p11, p12]]
                att_probs = self.species_attributes[species][att]
                for i in range(3):
                    prob_sum[0][i] += priors[species] * att_probs[0][i]
                    prob_sum[1][i] += priors[species] * att_probs[1][i]
            self.attributes[att] = prob_sum

        # Set the current distribution to the priors
        self.species_dist = self.priors[:]

    def prob(self, questions):
        '''Returns the probability p(q_1, ..., q_k) of the given questions.
        questions is a list of [question, [answer, confidence]].
        '''
        # Assume independence given the bird.
        # Sum over birds
        # TODO: Make this faster.  This slows down the overall algorithm by
        # a factor of roughly 2.
        sum_prob = 0.
        for species in range(nspecies):
            prob = 1.
            for [q, [a, c]] in questions:
                # q is an attribute number [0, 287]
                # a is 0 or 1
                # c is 0, 1, or 2
                prob *= self.species_attributes[species][q][a][c]
            sum_prob += prob * self.species_dist[species]
        return sum_prob


    def cond_prob(self, questions, species):
        '''Returns the probability p(q_1, ..., q_k | s) of the given questions.
        questions is a list of [question, [answer, confidence]], and species is
        an integer in (0, 199) giving the species of the bird.
        '''
        # Assume independence
        # p(q_1, ..., q_k | s) = p(q_1 | s) p(q_2 | s) ... p(q_k | s)
        prob = 1.
        for [q, [a, c]] in questions:
            prob *= self.species_attributes[species][q][a][c]
        return prob

    def reset_prob(self):
        '''
        Reset the distribution on s to the distribution on construction
        '''
        self.species_dist = self.priors[:]

    def cond_on_question(self, question):
        '''
        Modify the distribution on s by conditioning on the answer to a
        particular question
        '''
        q, [a, c] = question

        # By baye's rule, P(s | q) = P(q | s) * P(s) / P(q). The p(q)
        # is constant among all birds, so it is just a normalizing factor.
        cond_probs = [self.species_attributes[s][q][a][c]
                      * self.species_dist[s]
                      for s in range(nspecies)]
        sum_cond_probs = sum(cond_probs)
        self.species_dist = [val / sum_cond_probs for val in cond_probs]

    def remove_bird(self, bird):
        '''
        Remove a bird from consideration
        '''
        self.species_dist[bird] = 0
        newSum = sum(self.species_dist)

        self.species_dist = [val / newSum for val in self.species_dist]

    def get_probs(self):
        '''
        Returns the current distribution on birds
        '''
        return self.species_dist

class independent_probability_diffuse(independent_probability):
    '''Represents a probability distribution where the attributes are all
    independent and regularization is done using a pseudo-diffusive procedure.
    '''

    # IMPLEMENTATION NOTES:
    # IF THE GIVEN COUNTS DON'T INCLUDE ANY INSTANCES OF A (SPECIES, ATTRIBUTE)
    # PAIR, THOSE PROBABILITIES ARE SET TO 0.


    def __init__(self, counts, priors):
        '''Initializes this probability distribution, given a dictionary of
        counts.  The keys of counts take the form (species, attribute), and
        the values are [[c00, c01, c02], [c10, c11, c12]], where cij is the
        count of answer i with certainty j.  species and attribute are
        0-indexed.  priors is a list of prior probabilities.
        '''
        self.priors = priors

        # Get probabilities p(q | s)
        # Probabilities are indexed by [species][attribute]
        self.species_attributes = [[None] * nattributes for _ in
                                   range(nspecies)]
        for key in counts.keys():
            (species, attribute) = key
            [c0, c1] = counts[key]
            total_count = float(sum(c0) + sum(c1))
            total_count = max(total_count, 1)
            p0 = [c / total_count for c in c0]
            p1 = [c / total_count for c in c1]
            # Do the diffusive regularization
            # 0 is definite, 1 is confident, 2 is don't know
            # TODO: Verify those interpretations are correct
            # So make diffusion occur from
            # (present) 0 <-> 1 <-> 2 <-> (not present) 2 <-> 1 <-> 0
            # TODO: Make this depend on total_count
            # Maybe divide the multiplier by sqrt(total_count) or something
            # like that.
            probs = [p0[0], p0[1], p0[2], p1[2], p1[1], p1[0]]
            for _ in xrange(num_diffusions):
                new_probs = [0.] * 6
                for i in range(6):
                    # Special case the two certain cases
                    if i == 0:
                        new_probs[i + 1] += diffuse_multiplier * probs[i]
                    elif i == len(probs) - 1:
                        new_probs[i - 1] += diffuse_multiplier * probs[i]
                    else:
                        new_probs[i - 1] += .5 * diffuse_multiplier * probs[i]
                        new_probs[i + 1] += .5 * diffuse_multiplier * probs[i]
                    # Retain the rest of the probability
                    new_probs[i] += (1. - diffuse_multiplier) * probs[i]
                probs = new_probs;
            p0 = [probs[0], probs[1], probs[2]]
            p1 = [probs[5], probs[4], probs[3]]

            self.species_attributes[species][attribute] = [p0, p1]

        # Get probabilities p(q) = sum_s p(s) p(q | s)
        self.attributes = [None] * nattributes
        for att in xrange(nattributes):
            prob_sum = [[0., 0., 0.], [0., 0., 0.]]
            for species in xrange(nspecies):
                # [[p00, p01, p02], [p10, p11, p12]]
                att_probs = self.species_attributes[species][att]
                for i in range(3):
                    prob_sum[0][i] += priors[species] * att_probs[0][i]
                    prob_sum[1][i] += priors[species] * att_probs[1][i]
            self.attributes[att] = prob_sum

        
        # Set the current distribution to the priors
        self.species_dist = self.priors[:]


if __name__ == '__main__':
    asker = BayesAsker_A10()
    asker.init()
    #asker.probability = dummy_probability(None, None)
    import time
    start_time = time.time()
    print asker.myAvianAsker('image', [])
    print "time taken to ask question: ", time.time() - start_time
    import cProfile
    asker.init()
    cProfile.run('asker.myAvianAsker("image", [])')
