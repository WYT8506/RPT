import pandas as pd
import numpy as np
import copy

class read:
    def readChance(data):
        chanceList = []
        matrix3d = []
        for i in range(386):
            matrix3d.append([])
            for j in range(3):
                matrix3d[i].append([])
                for k in range(1):
                    m = i * 3 + j
                    tempList = []

                    datastore0 = data.loc[[m], ['SURVIVAL']]
                    datatrans0 = datastore0.values.tolist()
                    p = ''.join(datatrans0[0])
                    splited = p.split(',')

                    tempList.append(splited)
                    #print(splited[0][9:])
                    w = int(splited[0][9:])/100
                    wo = int(splited[1][12:splited[1].index('}')])/100
                    chanceList.append([w,wo])

        #print(chanceList)
        return chanceList



    def readAttr(data):
        attrList = []

        allCareer = data.loc[:, ['CAREER']]
        allCareerList = allCareer.values.tolist()
        list0 = []
        for element in allCareerList:
            element = ''.join(map(str, element))
            list0.append(element)
        setOfCareer = set(list0)
        listOfCareer = list(setOfCareer)
        len1 = len(listOfCareer)
        #print(len(listOfCareer))


        allPurpose = data.loc[:, ['PURPOSE OF TRIP']]
        allPurposeList = allPurpose.values.tolist()
        list00 = []
        for element in allPurposeList:
            element = ''.join(map(str, element))
            list00.append(element)
        setOfPurpose = set(list00)
        listOfPurpose = list(setOfPurpose)
        len2 = len(listOfPurpose)
        # print(len(listOfPurpose))

        # data_matrix[1] = datastore
        matrix3d = []

        for i in range(386):
            matrix3d.append([])
            for j in range(3):
                matrix3d[i].append([])
                for k in range(1):
                    m = i * 3 + j
                    tempList = []


                    datastore1 = data.loc[
                        [m], ['AGE']]
                    agebuf = datastore1.values.tolist()
                    if (agebuf[0][0] == 'young child'):
                        agebuf[0][0] = 0
                    elif (agebuf[0][0] == 'college student'):
                        agebuf[0][0] = 2
                    elif (agebuf[0][0] == 'middle aged' or agebuf[0][0] == '27 year old'):
                        agebuf[0][0] = 3
                    elif (agebuf[0][0] == 'senior citizen'):
                        agebuf[0][0] = 4
                    elif (0 < int(agebuf[0][0]) <= 5):
                        agebuf[0][0] = 0
                    elif (18 < int(agebuf[0][0]) <= 24):
                        agebuf[0][0] = 2
                    elif (24 < int(agebuf[0][0]) <= 50):
                        agebuf[0][0] = 3
                    tempList.append(agebuf[0][0])
                    # print(tempList)

                    datastore2 = data.loc[
                        [m], ['GENDER']]
                    genderbuf = datastore2.values.tolist()

                    if (genderbuf[0][0] == 'male' or genderbuf[0][0] == 'man'):
                        genderbuf[0][0] = 0
                    elif (genderbuf[0][0] == 'female'):
                        genderbuf[0][0] = 1
                    else:
                        genderbuf[0][0] = 0.5
                    tempList.append(genderbuf[0][0])



                    datastore3 = data.loc[
                        [m], ['HEALTH']]
                    health_buf = datastore3.values.tolist()

                    if (health_buf[0][0] == 'in great health'):
                        health_buf[0][0] = 1
                    elif (health_buf[0][0] == 'with asthma'):
                        health_buf[0][0] = 2
                    elif (health_buf[0][0] == 'who is wheelchair bound'):
                        health_buf[0][0] = 3
                    elif (health_buf[0][0] == 'who is terminally ill with 5 years left'):
                        health_buf[0][0] = 4

                    else:
                        health_buf[0][0] = 0
                    tempList.append(health_buf[0][0])



                    datastore4 = data.loc[
                        [m], ['CAREER']]
                    career_buf = datastore4.values.tolist()
                    index = m % len1

                    if (career_buf[0][0] == listOfCareer[index]):
                        career_buf[0][0] = 1
                    else:
                        career_buf[0][0] = 0
                    tempList.append(career_buf[0][0])

                    datastore5 = data.loc[
                        [m], ['PURPOSE OF TRIP']]
                    purpose_buf = datastore5.values.tolist()
                    # print(len(listOfCareer))
                    index = m % len2

                    if (purpose_buf[0][0] == listOfPurpose[index]):
                        purpose_buf[0][0] = 1
                    else:
                        purpose_buf[0][0] = 0
                    tempList.append(purpose_buf[0][0])

                    attrList.append(tempList)
        #print(attrList)


        return  attrList


    def readRank(data):
        ranklist = []
        matrix3d2 = []

        for i in range(386):
            matrix3d2.append([])
            for j in range(3):
                matrix3d2[i].append([])
                for k in range(1):
                    m = i * 3 + j
                    datastore6 = data.loc[
                        [m], ['Ranking']]
                    ranking_buf = datastore6.values.tolist()
                    A = ranking_buf[0][0].index('A') / 2
                    B = ranking_buf[0][0].index('B') / 2
                    C = ranking_buf[0][0].index('C') / 2

                    ranklist1 = []
                    ranklist1.append(int(A))
                    ranklist1.append(int(B))
                    ranklist1.append(int(C))
                    ranklistTemp = ranklist1.copy()
                    rankList2 = []

                    if '=' in ranking_buf[0][0]:
                        if ranking_buf[0][0].count('=') == 2:
                            choiceList = [0, 1, 2]
                            A = np.random.choice(choiceList)
                            choiceList.remove(A)
                            B = np.random.choice(choiceList)
                            choiceList.remove(B)
                            C = np.random.choice(choiceList)
                            rankList2.append(int(A))
                            rankList2.append(int(B))
                            rankList2.append(int(C))
                            ranklist.append(rankList2)

                            # print(A,B,C)
                        elif ranking_buf[0][0].count('=') == 1:
                            if ranking_buf[0][0].index('=') == 1:

                                first = ranklistTemp.index(0)
                                second = ranklistTemp.index(1)
                                #print(ranklistTemp)
                                #print(ranklistTemp,1,ranklist1,first)
                                #print(ranklistTemp[first])
                                ranklistTemp[first] = np.random.choice([0,1])
                                #print(ranklistTemp[first])
                                #print(ranklist1)
                                if ranklistTemp[first] != ranklist1[first]:
                                    #print('happen')
                                    ranklistTemp[second] = ranklist1[first]
                                    #print(ranklistTemp)
                                #tieBreakTemp.append(ranklistTemp[second])
                                #print(ranklistTemp)
                                rankList2.append(ranklistTemp[0])
                                rankList2.append(ranklistTemp[1])
                                rankList2.append(ranklistTemp[2])
                                ranklist.append(rankList2)

                            else:
                                second = ranklistTemp.index(1)
                                third = ranklistTemp.index(2)
                                #print(ranklistTemp)
                                # print(ranklistTemp,1,ranklist1,first)
                                # print(ranklistTemp[first])
                                ranklistTemp[second] = np.random.choice([2, 1])
                                # print(ranklistTemp[first])
                                # print(ranklist1)
                                if ranklistTemp[second] != ranklist1[second]:
                                    #print('happen')
                                    ranklistTemp[third] = ranklist1[second]
                                    #print(ranklistTemp)
                                rankList2.append(ranklistTemp[0])
                                rankList2.append(ranklistTemp[1])
                                rankList2.append(ranklistTemp[2])
                                ranklist.append(rankList2)
                    else:
                        rankList2.append(int(A))
                        rankList2.append(int(B))
                        rankList2.append(int(C))
                        ranklist.append(rankList2)

        #print(ranklist)

        #print(rankList2)
        return ranklist



    def read_raw_data(file_name):
        rawData = pd.read_csv(file_name)
        return rawData
    """
    rankList = readRank(rawData)
    attrList = readAttr(rawData)
    chanceList = readChance(rawData)
    #print(rankList)
    print(attrList)
    #print(chanceList)
    """

