from surprise import accuracy
from collections import defaultdict
from statistics import mean
import math


class RecommenderMetrics:
    # Mean Absolute Error
    def MAE(self, predictions):
        return accuracy.mae(predictions, verbose=False)

    # Root Mean Square Error
    def RMSE(self, predictions):
        return accuracy.rmse(predictions, verbose=False)

    # Use parameter congruence to measure how similar the predictions are to user's preference, which is measured by the keywords
    # and abstracts of users' advisors' and coworkers' papers
    def GetTopN(self, predictions, n=10, miniCong=6):
        topN = defaultdict(list)

        for userID, paperID, actualCong, estimatedCong, _ in predictions:
            if estimatedCong >= miniCong:
                topN[int(userID)].append((int(paperID), estimatedCong))

        for userID, congruence in topN.items():
            congruence.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = congruence[:n]

        return topN

    # The following leftoutPrdictions comes form the application of leave-one-out cross validation
    def HitRate(self, topNPredicted, leftoutPredictions):
        hits = 0
        total = 0

        # For each left-out prediction
        for leftout in leftoutPredictions:
            userID = leftout[0]
            leftoutPaperID = leftout[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for paperID, predictedCong in topNPredicted[int(userID)]:
                if int(leftoutPaperID) == int(paperID):
                    hit = True
                    break
            if hit:
                hits += 1

            total += 1

        # compute overall precision
        return hits / total

    # Calculate the cumulative hit rate
    def CumulativeHitRate(self, topNPredicted, leftoutPredictions, congCutoff=0):
        hits = 0
        total = 0

        # For each left-out prediction
        for userID, leftoutPaperID, actualCong, estimatedCong, _ in leftoutPredictions:
            # Only look at ability to recommend things the users actually like
            if actualCong >= congCutoff:
                # Is it in the top 10 for this user?
                hit = False
                for paperID, predictedCong in topNPredictions[int(userID)]:
                    if int(leftoutPaperID) == int(paperID):
                        hit = True
                        break

                if hit:
                    hits += 1

            total += 1

        # Compute overall precision
        return hits / total

    # Now we need to rate the hit rate
    def RatingHitRate(self, topNPredicted, leftoutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out prediction
        for userID, leftoutPaperID, actualCong, estimatedCong, _ in leftoutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for paperID, predictedCong in topNPredicted[int(userID)]:
                if int(leftoutPaperID) == int(paperID):
                    hit = True
                    break
            if hit:
                hits[actualCong] += 1

            total[actualCong] += 1

        # Compute overall precision
        for cong in sorted(hits.keys()):
            print(cong, hits[cong] / total[cong])

    # Another metric for accuracy
    def AverageReciprocalHitRank(self, topNPredicted, leftoutPredictions):
        summation = 0
        total = 0

        # For each left-out prediction
        for userID, leftoutPaperID, actualCong, estimatedCong, _ in leftoutPredictions:
            # Is it in the predicted 10 for the user?
            hitRank = 0
            rank = 0
            for paperID, predictedCong in topNPredicted[int(userID)]:
                rank += 1
                if int(leftoutPaperID) == int(paperID):
                    hitRank = rank
                    break

            if hitRank > 0:
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(self, topNPredicted, numUsers, congThreshold=0):
        hits = 0

        for userID in topNPredicted.keys():
            hit = False
            for paperID, predictedCong in topNPredicted[int(userID)]:
                if predictedCong >= congThreshold:
                    hit = True
                    break

            if hit:
                hits += 1

        return hits / numUsers

    # Diversity = 1 - S, where S is the average similarity between recommendation pairs
    def Diversity(self, topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                paper1 = pair[0][0]
                paper2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(paper1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(paper2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return 1 - S

    # Novelty measures the mean popularity rank of recommended papers
    def Novelty(self, topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for cong in topNPredicted[userID]:
                paperID = cong[0]
                rank = rankings[paperID]
                total += rank
                n += 1

        return total / n

    # Above metrics measure the accuracy of papers in the list
    # Now we focus on the ordering of the papers in the list
    # In the follwoing test is a dictionary with values being paperID
    def mAP(self, topNPredicted, test):
        user_mAP = {}
        for userID in topNPredicted.keys():
            hits = 0
            rank = 0
            precision = []
            for paperID, _ in topNPredicted[userID]:
                if paperID in test[userID]:
                    hits += 1
                rank += 1
                precision.append(hits / rank)
            user_mAP[userID] = mean(precision)

        return user_mAP

    # Mean Reciprocal Rank quantifies the rank of the first relevant item found in the recommendation list.
    def MRR(self, topNPrdicted, test):
        MRR = {}
        for userID in topNPredicted.keys():
            index = 0
            found = False
            for paperID, _ in topNPredicted[userID]:
                index += 1
                if paperID in test[userID]:
                    MRR[userID] = 1.0 / index
                    found = True
                    break
                if not found:
                    MRR[userID] = 0

        return MRR

    # Cumulative Gain (CG) is the sum of relevant items among top n results.
    # Discounted Cumulative Gain (DGC) discounts the "value" of the relevant items based on its rank.
    # Normalized DCG (NDCG) normalizes DCG by the "ideal" recommendation algorithm.
    def NDCG(self, topNPredicted, test):
        NDCG = {}
        for userID in topNPredicted.keys():
            DCG = 0
            IDCG = 0
            for rank, (paperID, _) in enumerate(topNPredicted[userID], 1):
                if paperID in test[userID]:
                    DCG += 1 / math.log(rank + 1)
                if rank <= len(test[userID]):
                    IDCG += 1 / math.log(rank + 1)
            NDCG[userID] = DCG / IDCG if IDCG > 0 else 0

        return NDCG
