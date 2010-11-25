import cPickle
import math
import os

nspecies = 200
nattributes = 288
nattrvals = 6

class ClusterAsker_A10():

    def __init__(self):
        """
        Initializer for class instance
        """
        # Load up the bird averages
        stat_filename = "birdStats.dat"

        if os.path.exists(stat_filename):
            f = open(stat_filename, 'r')
            birdStats = cPickle.load(f)
            f.close()
        else:
            genenerateBirdStats(stat_filename)

        self.birdAttrs = birdStats[0]
        self.birdAttrVars = birdStats[1]

        # Pre-calculate some conditional probabilities of attributes
        # given a bird.
        self.birdAttrP = {}

        attrVals = [-1.0, -.66, -.33, .33, .66, 1.0]

        for b in range(nspecies):
            thisBirdAttrP = {}
            for attr in range(nattributes):
                vals = [math.exp(-(a - self.birdAttrs[b][attr]) ** 2
                                   / self.birdAttrVars[b][attr])
                                   for a in attrVals]

                vals = [vals[i] / sum(vals)
                        for i in range(nattrvals)]

                thisBirdAttrP[attr] = vals
            self.birdAttrP[b] = thisBirdAttrP
        
        f.close()

    def init(self):
        """
        Start up a new instance of a bird to guess
        """
        # Set the initial weights on each bird
        self.weights = [1.0 / nspecies] * nspecies

        # Set squared norms for each bird. Start it off at some small
        # value to avoid divide by 0
        self.norms = [0.01] * nspecies
 
        # Set the answers for each question so far
        self.pastAnswered = set()

    def myAvianAsker(self, image, questions):
        """
        Get the set of past questions and return a new question to
        ask.

        NOTE: this function assumes that calling conventions of the
        testbed, so the set of questions must be asked in order, and
        only after init is used can a new set of questions be asked.
        """
        if len(questions) == 0:
            # No previous questions asked, no need to update weights before
            # asking.
            ask, score = self.pickBestAttrQuestion()
            print "Asking ", ask
            print "Score is ", score
            return ask

        # Extract the last question asked. It is assumed that previous
        # questions were already processed.
        last = questions[-1]

        question = last[0]
        answer = last[1]

        # Update the record of past answered
        self.pastAnswered.add(question)

        # Due to some wierd coding, the type of answer can vary from
        # '0', '2', or [has, sure] tuples.
        if answer == '0':
            # This occurs if the bird was guessed and wrong. In this
            # case, the particular bird should be removed from consideration.
            # One way to do this is to massively increase the distance of
            # the incorrect bird
            self.norms[question - nattributes] += 1000000.0
        elif answer == '2':
            pass
        else:
            if answer == '2':
                # This occurs if a question had no data to it. In this case,
                # the answer is almost the same as if there is no data in
                # the training set, which would lead to a linear value of 0
                val = 0.0 
            else:
                # A tuple of has, sure.
                has = answer[0]
                sure = answer[1]
                val = (has * 2.0 - 1.0) * ((2.0 - sure) / 3.0 + 1.0 / 3.0)

            # Update the squared distances for each bird. Each squared distance
            # is modified by how variable that particular attribute is for
            # the bird.
            self.norms = [self.norms[i] +
                          (val - self.birdAttrs[i][question]) ** 2
                          / self.birdAttrVars[i][question]
                          for i in range(nspecies)]

        # Update weights with a negative exponential. This is proportional
        # to the probability of a guassian with non-correlated inputs, as
        # the norms have been calculated by dividing the variances at each
        # attribute.
        self.weights = [math.exp(-self.norms[i] / 2.0)
                        for i in range(nspecies)]
        weightSum = sum(self.weights)
        self.weights = [self.weights[i] / weightSum for i in range(nspecies)]

        # Get the best attribute
        askAttr, score = self.pickBestAttrQuestion()
        birdGain = self.bestBirdEntropyGain()
        print "Best Attr is", askAttr
        print "Score is", score
        print "Bird entropy gain", birdGain

        # Choose the best entropy reducing question, the attribute or bird
        if birdGain > score:
            ask = self.pickBestBird()
            print "Asking bird", ask
            print "Best bird norms are", sorted(self.norms)[:5],\
                                         sorted(self.norms)[nspecies-3:]
            return ask
        else:
            print "Asking attr", askAttr
            print "bird norms are", sorted(self.norms)[:5],\
                                    sorted(self.norms)[nspecies-3:]
            return askAttr

    def pickBestAttrQuestion(self):
        """
        Returns the best attribute question to ask that is not already asked,
        along with the score (mutual information) that the question has
        """
       
        # First calculate(?) the probability of each attribute value 
        attrValsP = {}
        for attr in range(nattributes):
            probs = [0] * nattrvals

            for a in range(nattrvals):
                for b in range(nspecies):
                    probs[a] += self.birdAttrP[b][attr][a] * self.weights[b]

            attrValsP[attr] = probs

        # Calculate the mutual info
        scores = [0.0] * nattributes
        for attr in range(nattributes):
            for b in range(nspecies):
                for a in range(nattrvals):
                    pAB = self.birdAttrP[b][attr][a] * self.weights[b]
                    if pAB != 0.0:
                        logPart = pAB / (attrValsP[attr][a] * self.weights[b])
                        scores[attr] += pAB * math.log(logPart)

            scores[attr] /= math.log(2)

        # Now get the best score 
        bestScore = 0.0
        bestI = 0
        for i in range(nattributes):
            score = scores[i]
            if i not in self.pastAnswered and score > bestScore:
                bestScore = score
                bestI = i

        return bestI, bestScore

    def bestBirdEntropyGain(self):
        """
        Returns the expected entropy gain when the best bird is guessed.
        """
        bestWeight = max(self.weights)

        # Get the full entropy currenty
        fullEntropy = sum([-self.weights[b] * math.log(self.weights[b])
                            if self.weights[b] > 0 else 0 
                            for b in range(nspecies)]) / math.log(2)
 
        # Get the entropy assuming the bird was wrong and given a weight of
        # 0.
        newWeightSum = sum(self.weights) - bestWeight
        newWeights = [self.weights[b] / newWeightSum for b in range(nspecies)]
        newEntropy = sum([-newWeights[b] * math.log(newWeights[b])
                            if newWeights[b] > 0 else 0
                            for b in range(nspecies)])
        newEntropy += bestWeight/newWeightSum * \
                      math.log(bestWeight/newWeightSum)

        newEntropy /= math.log(2)

        # Get the entropy gain if the bird is wrong
        gain = fullEntropy - newEntropy

        # If the bird is guess correctly, all of the entropy is removed.
        return bestWeight * fullEntropy + (1 - bestWeight) * gain

    def pickBestBird(self):
        """
        Returns the question referring to the best bird that has not already
        been asked
        """
        bestWeight = 0.0
        bestI = 0
        for i in range(nspecies):
            if i + nattributes not in self.pastAnswered \
               and self.weights[i] > bestWeight:
                bestWeight = self.weights[i]
                bestI = i + nattributes

        return bestI

def generateBirdStats(filename):
    f = open('student/student_dataset.txt', 'r')
    imageToBird = getImageToBird()

    birdCounts = {}

    nattributes = 288

    for i, line in enumerate(f):
        parts = line.split()

        image = int(parts[0])
        attr  = int(parts[1])
        has   = int(parts[2])
        sure  = int(parts[3])
        user  = int(parts[4])

        # Make the birds 0-indexed
        bird = imageToBird[image] - 1

        if bird not in birdCounts:
            birdCounts[bird] = []
            for i in range(nattributes):
                birdCounts[bird].append([0, 0, 0])

        modHas = (has * 2.0 - 1.0) * ((2.0 - sure) / 3.0 + 1.0 / 3.0)

        # The first is a sum, to get the means
        birdCounts[bird][attr][0]  += modHas

        # The second is a sum of squares, to get the second moment, and
        # later the variance
        birdCounts[bird][attr][1]  += modHas * modHas

        # The third is the total number of records
        birdCounts[bird][attr][2]  += 1

        if i % 100000 == 0:
            print i

    birdStats = {}
    meanMap = {}
    varMap = {}
    for b in birdCounts:
        counts = birdCounts[b]

        extraRecsPer = 5

        totalExtraRecs = extraRecsPer * 6
        extraSquares = 3.111111
        # Get the mean, regularized by assuming 5 records of each of
        # the 6 possible
        # response is present. Note that the mean of these extra is 0.
        means = [counts[i][0] / (counts[i][2] + totalExtraRecs)
                    for i in range(nattributes)]

        # Get the variance. Again, the same above records are added.
        var = [(counts[i][1] + extraSquares) / (counts[i][2] + totalExtraRecs)
                     - means[i] ** 2
               for i in range(nattributes)]

        meanMap[b] = means
        varMap[b] = var

    f.close()

    # Save it up
    birdStats = [meanMap, varMap]
    f = open(filename, "w")
    cPickle.dump(birdStats, f)
    f.close()

def getImageToBird():
    """
    Returns a dictionary mapping image id to bird id
    """
    f = open('student/student_image.txt', 'r')

    imageToBird = {}

    for line in f:
        parts = line.split()

        image = int (parts[0])
        bird  = int (parts[1].split('.')[0])
        imageToBird[image] = bird

    return imageToBird
