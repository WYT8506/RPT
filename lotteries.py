import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import random
from random import choices
from scipy import stats
from scipy.stats import norm
from scipy import integrate
import numpy as np
from numpy import linspace, sum
import math
from sympy import *
import copy
random.seed(a=None, version=2)
class ground_truth:
    def __init__(self):
        return
    def set_beta(self,beta):
        self.beta = beta
    def get_beta(self):
        return self.beta
    def set_phi(self,phi):
        self.phi = phi
    def get_phi(self):
        return self.phi

class person:
    def __init__(self,index, feature_values, p_with, p_without):
        self.index = index
        self.feature_values = feature_values
        self.p_with = p_with
        self.p_without = p_without
    def get_p_with(self):
        return self.p_with
    def get_p_without(self):
        return self.p_without
    """
    def get_utility(self,beta):
        if len(self.feature_values) != len(beta):
            print("error")
            return 
        return np.dot(np.array(self.feature_values),np.array(beta))
    """
    def get_features(self):
        return self.feature_values
class data_structure:
    def __init__(self,person_matrix): #person matrix: number of questions * number of lotteries in each question
            self.person_matrix = person_matrix
            self.number_of_questions = len(person_matrix)
            self.outcomes = []
            for i in range(2**len(person_matrix[0])):
                binary_string = bin(i)[2:]
                binary_string = binary_string.rjust(len(person_matrix[0]),'0')
                self.outcomes.append(binary_string)
            self.outcome_probabilities_matrixs = []##number of questions * number of lotteries * number of outcomes
            self.outcome_feature_matrixs = []#number of questions * number of outcomes * number of features for any outcome
            self.counter_matrixs = []# this is a data structure to store preference profiles. It's size is: number of questions * number of lotteries * number of lotteries

            self.initialize_outcome_feature_matrixs()
            self.initialize_probability_matrixs()
            #add a outcome_probabilities list into outcome_probabilities_matrix
    def get_person_matrix(self):
        return self.person_matrix
    def get_number_of_questions(self):
        return self.number_of_questions
    def get_outcomes(self):
        return self.outcomes
    def get_counter_matrixs(self):
        return self.counter_matrixs
    def get_outcome_feature_matrixs(self):
        return self.outcome_feature_matrixs
    def get_outcome_probabilities_matrixs(self):
        return self.outcome_probabilities_matrixs
    def initialize_outcome_feature_matrixs(self):
        for n in range(len(self.person_matrix)):
            outcome_features = []
            for i in range(len(self.outcomes)):
                outcome = self.outcomes[i]
                feature_vector = np.array([0]*len(self.person_matrix[0][0].get_features()))
                for j in range(len(outcome)):
                        if outcome[j]=='1':
                            feature_vector=np.array(self.person_matrix[n][j].get_features())+feature_vector
                outcome_features.append(feature_vector.tolist())
            self.outcome_feature_matrixs.append(outcome_features)
    def initialize_probability_matrixs(self):  
        for question_number in range(len(self.person_matrix)):   
            outcome_probabilities_matrix = []
            for chosen_person_index in range(len(self.person_matrix[0])):
                outcome_probabilities = []
                for i in range(len(self.outcomes)):
                    outcome = self.outcomes[i]
                    P = 1
                    for j in range(len(outcome)):
                        if j!= chosen_person_index:
                            if outcome[j] == '1':
                                P=P*(self.person_matrix[question_number][j].get_p_without())
                            else:
                                P=P*(1-self.person_matrix[question_number][j].get_p_without())
                        else:
                            if outcome[j] == '1':
                                P=P*(self.person_matrix[question_number][j].get_p_with())
                            else:
                                P=P*(1-self.person_matrix[question_number][j].get_p_with())
                    outcome_probabilities.append(P)
                outcome_probabilities_matrix.append(outcome_probabilities)
            self.outcome_probabilities_matrixs.append(outcome_probabilities_matrix)


    def build_counter_matrixs(self,preference_profiles):
        if (len(preference_profiles)!= self.number_of_questions):
            print("profile size wrong")
        for preference_profile in preference_profiles:
            num_candidates = len(preference_profile[0])
            counter_matrix = []
            for i in range(num_candidates):
                l = []
                for j in range(num_candidates):
                    l.append([(i,j),0])
                counter_matrix.append(l)

            for ranking in preference_profile:
                for i in range(num_candidates):
                    for j in range(num_candidates):
                        if ranking.index(i)<ranking.index(j):
                            counter_matrix[i][j][1]+=1
            self.counter_matrixs.append(counter_matrix)
     
class data_generator:
        # a data generator combines our data structure with a specific beta and phi 
        def __init__(self,data_structure,beta,phi):
            self.outcome_utility_vectors = []# number of questions * number of outcomes
            self.phi = phi
            self.beta = beta
            self.means_matrix = []
            self.initialize_outcome_utility_vectors(data_structure)
            self.initialize_means_matrix(data_structure)
            #self.distributions = []
        def get_outcome_utility_vectors(self):
            return self.outcome_utility_vectors
        def get_means_matrix(self):
            return self.means_matrix

        def initialize_outcome_utility_vectors(self,data_structure):
            outcomes = data_structure.get_outcomes()
            outcome_feature_matrixs = data_structure.get_outcome_feature_matrixs()
            number_of_questions = data_structure.get_number_of_questions()
            for question_number in range(number_of_questions):
                outcome_utilities = []
                for i in range(len(outcomes)):
                    outcome = outcomes[i]
                    utility = np.dot(np.array(outcome_feature_matrixs[question_number][i]),np.array(self.beta))
                    outcome_utilities.append(utility)
                self.outcome_utility_vectors.append(outcome_utilities)


        def p_reweighting(self,p,phi):
            gamma = phi[0]
            return p**gamma/((p**gamma+(1-p)**gamma)**(1/gamma))
            """
            beta = phi[1]
            alpha = phi[0]
            return exp(-beta*(-np.log(p))**alpha)
            """

        def initialize_means_matrix(self,data_structure):
            outcomes = data_structure.get_outcomes()
            outcome_probabilities_matrixs = data_structure.get_outcome_probabilities_matrixs()
            #print(outcome_probabilities_matrixs)
            for question_number in range(data_structure.get_number_of_questions()):
                outcome_probabilities_matrix = outcome_probabilities_matrixs[question_number]
                means = []
                number_of_lotteries = len(outcome_probabilities_matrix)
                for i in range(number_of_lotteries):
                    sum_pi_v = 0
                    sum_pi = 0
                    sum_pi_square = 0
                    for j in range(len(self.outcome_utility_vectors[0])):
                        u = self.outcome_utility_vectors[question_number][j]
                        p = outcome_probabilities_matrix[i][j]
                        #if self.phi == 0.1 :
                            #print(u,self.p_reweighting(p,self.phi))
                        sum_pi_v += u*self.p_reweighting(p,self.phi)
                        sum_pi += self.p_reweighting(p,self.phi)
                        #sum_pi_square += self.p_reweighting(p)**2
                    means.append(sum_pi_v/sum_pi)
                self.means_matrix.append(means)
                #self.standard_deviation = (self.sd**2*sum_pi_square/(sum_pi**2))**(1/2)
                #self.distributions.append(stats.norm(self.means[i],self.standard_deviation))
        def log_p_j1_j2(self,index,j1,j2):
            means = self.means_matrix[index]
            """
            print("beta is",self.beta)
            print("outcome utilities",self.outcome_utilities)
            print("means",self.means)
            print("cll is",math.log(exp(self.means[j1])/(exp(self.means[j1])+exp(self.means[j2]))))
            """
            return math.log(exp(means[j1])/(exp(means[j1])+exp(means[j2])))

        def p_j1_j2(self,index,j1,j2):
            means = self.means_matrix[index]
            """
            print("beta is",self.beta)
            print("outcome utilities",self.outcome_utilities)
            print("means",self.means)
            print("cll is",math.log(exp(self.means[j1])/(exp(self.means[j1])+exp(self.means[j2]))))
            """
            return exp(means[j1])/(exp(means[j1])+exp(means[j2]))
        def sample_a_ranking(self,index):
            lotteries = []
            weights = []
            ranking = []
            
            means = self.means_matrix[index]
            #print(self.outcome_utilities)
            #print(self.outcome_probabilities_matrix)
            #print(self.means)
            for i in range(len(means)):
                lotteries.append(i)
            for i in range(len(means)):
                weights.append(math.exp(means[i]))
            for i in range(len(lotteries)):
                weights_sum = sum(weights)
                for j in range(len(weights)):
                    weights[j] = weights[j]/weights_sum

                #print(lotteries)
                #print(weights)
                winner = np.random.choice(np.arange(0,len(lotteries)),p=weights)
                #print(winner)
                ranking.append(winner)
                weights[winner]=0
            #print(ranking)
            return ranking

  #     def get_average_prospect(self,lottery_index):
   #        return random.gauss(self.means[lottery_index],self.standard_deviation)


        #def get_PDF(self,lottery_index, x):
         #   return self.distributions[lottery_index].pdf(x)
            




"""
def midpoint_double1(f, lower,upper, n):
    h = (upper-lower)/float(n)
    I = 0
    for i in range(n):
        for j in range(i,n):
            xj1 = lower + h/2 + i*h
            xj2 = lower + h/2 + j*h
            I += h**2*f(xj1,xj2)
            #print((xj1,xj2,I))
    return I
def R_probability(lotteries_ranking,U,u_alpha,u_beta,u_phi,u_rf):
    global lotteries
    lotteries = []
    for lottery in lotteries_ranking:
        lottery.set_utilities(U)
        lottery.set_u_parameters(u_alpha,u_beta,u_phi,u_rf)
        lottery.set_distribution()
        lotteries.append(lottery)
    print('dfa')
    return midpoint_double1(f,-10,10,100)
    #return integrate.nquad(f, [bounds_xj2])

def f(xj1,xj2):
    #print(len(lotteries))
    return lotteries[0].PDF(xj1)*lotteries[1].PDF(xj2)
def bounds_xj4():
    return [-100,100]
def bounds_xj3(xj4):
    return [xj4,1]
def bounds_xj2():
    return [-1000,1000]
def bounds_xj1():
    return [-1000,1000]
"""
class algorithm:
    def algorithm1(data_structure,beta_initial,phi_range,tolerance,learning_rate):
        max_cll = -10000
        best_beta =None
        best_phi = None
        for t in range(0,int((phi_range[1]-phi_range[0])/tolerance)):
            phi = phi_range[0]+t*tolerance
            beta, cll= algorithm.gradient_decent(data_structure,beta_initial,[phi],learning_rate)
            #find_beta_thatmaximizecll
            if (cll >max_cll):
               max_cll = cll
               best_beta = beta
               best_phi = [phi]
            print("cll for ",phi)
            print(cll)
            print("beta is ")
            print(beta)
        return (best_beta,best_phi)

    def gradient_decent(data_structure,beta_initial,phi,learning_rate):
        beta = copy.deepcopy(beta_initial)
        for i in range(3000):
            d_cll = algorithm.d_beta_cll(data_structure,beta,phi)#naive method for reference
            d_cll1 =[0]*len(beta) 
            for question_number in range(data_structure.get_number_of_questions()):
                d = algorithm.d_beta_cll1(data_structure,question_number,beta,phi)
                d_cll1 = list(np.array(d)+np.array(d_cll1))
            if random.choice([0,1,2,3,4,5,6,7,8,9,10]) == 0:
                print("d_cll:",d_cll1) 

            if i == 2999:
                print("final_d:",d_cll1)
            for j in range(len(beta)):
                beta[j]=beta[j]+d_cll1[j]*learning_rate
        final_cll = algorithm.cll(data_structure,beta,phi)
        return (copy.deepcopy(beta),final_cll)



    def d_beta_cll(data_structure,beta,phi):
        vector = []
        h = 1e-8
        for i in range(len(beta)):
            new_beta = copy.deepcopy(beta)
            new_beta[i] = beta[i]+h
            vector.append((algorithm.cll(data_structure,new_beta,phi)-algorithm.cll(data_structure,beta,phi))/h)
        return vector

    def d_beta_cll1(data_structure,index,beta,phi):#index means the question number
        vector = []
        dg = data_generator(data_structure,beta,phi)
        counter_matrix = data_structure.get_counter_matrixs()[index]
        num_candidates = len(counter_matrix)

        for i in range(len(beta)):
            beta_i = beta[i]
            d_cll = 0
            for j in range(num_candidates):
                for k in range(num_candidates):
                    if counter_matrix[j][k][1]>0:
                          d_cll+=(counter_matrix[j][k][1]/(counter_matrix[k][j][1]+counter_matrix[j][k][1])/dg.p_j1_j2(index,j,k))*algorithm.d_p_j1_j2(index,j,k,phi,i,beta,data_structure,dg)
            vector.append(d_cll)

        return vector
    
    def d_p_j1_j2(index,j1,j2,phi,beta_index,beta,data_structure,dg):
        outcome_features = data_structure.get_outcome_feature_matrixs()[index]
        outcome_probabilities_matrix = data_structure.get_outcome_probabilities_matrixs()[index]
        p_normalize1 = 0 
        p_normalize2 = 0

        for i in range(len(outcome_features)):
            p_normalize1+=dg.p_reweighting(outcome_probabilities_matrix[j1][i],phi)
            p_normalize2+=dg.p_reweighting(outcome_probabilities_matrix[j2][i],phi)
        #print(p_normalize1,p_normalize2)

        k1 = 0
        k2 = 0
        for i in range(len(outcome_features)):
            k1+=outcome_features[i][beta_index]*dg.p_reweighting(outcome_probabilities_matrix[j1][i],phi)
            k2+=outcome_features[i][beta_index]*dg.p_reweighting(outcome_probabilities_matrix[j2][i],phi)
        c1 = 0
        c2 = 0
        for i in range(len(outcome_features)):
            #print(np.dot(outcome_features[i],beta))
            c1+=(np.dot(outcome_features[i],beta)-outcome_features[i][beta_index]*beta[beta_index])*dg.p_reweighting(outcome_probabilities_matrix[j1][i],phi)
            c2+=(np.dot(outcome_features[i],beta)-outcome_features[i][beta_index]*beta[beta_index])*dg.p_reweighting(outcome_probabilities_matrix[j2][i],phi)

        k1 = k1/p_normalize1
        k2 = k2/p_normalize2
        c1 = c1/p_normalize1
        c2 = c2/p_normalize2

        #print(k1,k2,c1,c2)
        
        beta_i = beta[beta_index]
        beta_i1 = beta[beta_index]+1e-7
        #print("real means: ",k1*beta_i+c1,k2*beta_i +c2)
        g =exp(k1*beta_i+c1)
        h =exp(k1*beta_i+c1)+exp(k2*beta_i+c2)
        #g_ =exp(k1*beta_i1+c1)
        #h_ =exp(k1*beta_i1+c1)+exp(k2*beta_i1+c2)
        #print("suggested likelyhood change is:",((g/h)-(g_/h_))/1e-7)
        d_g = k1*exp(k1*beta_i+c1)
        d_h = k1*exp(k1*beta_i+c1)+k2*exp(k2*beta_i+c2)
        return (d_g*h-g*d_h)/(h**2)

    def cll(data_structure,beta,phi):
        total_cll = 0
        counter_matrixs = data_structure.get_counter_matrixs()
        dg = data_generator(data_structure,beta,phi)
        for m in range(len(counter_matrixs)):
            counter_matrix = counter_matrixs[m]
            CLL = 0
            num_candidates = len(counter_matrix)
            for i in range(num_candidates):
                for j in range(num_candidates):
                    if counter_matrix[i][j][1]>0:
                        CLL+=counter_matrix[i][j][1]/(counter_matrix[j][i][1]+counter_matrix[i][j][1])*dg.log_p_j1_j2(m,i,j)
            total_cll+=CLL
        #print(beta,phi,CLL)
        return CLL

class plot:
    def plot_beta_3d(data_structure,phi):
        phi = phi[0]
        X = []
        Y = []
        Z = []
        for i in range(-70,70):
            for j in range(-70,70):
                X.append(i*0.07)
                Y.append(j*0.07)
        max_z = -100000
        max_x = -10000
        max_y = -10000
        for i in range(len(X)):
            CLL = algorithm.cll(data_structure,[X[i],Y[i]],phi)
            Z.append(CLL)
            if CLL> max_z:
                max_z = CLL
                max_x = X[i]
                max_y = Y[i]
        print(max_x,max_y,max_z)
        #print(algorithm.cll(data_structure,preference_profile,[1,1],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[1.2,1],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[0.8,1],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[1,1.2],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[1,0.8],0.5))
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X,Y,Z,depthshade = 0.5) 
        # naming the x axis 
        #plt.xlabel('x - axis')
        # naming the y axis 
        #plt.ylabel('y - axis') 
        #plt.zlabel('z - axis') 
        # function to show the plot 
        plt.show()

    def plot_phi_3d(data_structure,beta):
        X = []
        Y = []
        Z = []
        for i in range(-29,29):
            for j in range(-29,29):
                X.append(i/60+0.5)
                Y.append(j/60+0.5)

        for i in range(len(X)):
            Z.append(algorithm.cll(data_structure,beta,[X[i],Y[i]]))
        #print(algorithm.cll(data_structure,preference_profile,[1,1],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[1.2,1],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[0.8,1],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[1,1.2],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[1,0.8],0.5))
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X,Y,Z,depthshade = 0.5) 
        # naming the x axis 
        #plt.xlabel('x - axis')
        # naming the y axis 
        #plt.ylabel('y - axis') 
        #plt.zlabel('z - axis') 
        # function to show the plot 
        plt.show()
    def plot_phi_2d(data_structure,preference_profile):
        X = []
        Y = []
        Z = []
        for j in range(-29,29):
            X.append(1)
            Y.append(j/60+0.5)

        for i in range(len(X)):
            Z.append(algorithm.cll(data_structure,preference_profile,[1,2],Y[i]))
        #print(algorithm.cll(data_structure,preference_profile,[1,1],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[1.2,1],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[0.8,1],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[1,1.2],0.5))
        #print(algorithm.cll(data_structure,preference_profile,[1,0.8],0.5))
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X,Y,Z,depthshade = 0.5) 
        # naming the x axis 
        #plt.xlabel('x - axis')
        # naming the y axis 
        #plt.ylabel('y - axis') 
        #plt.zlabel('z - axis') 
        # function to show the plot 
        plt.show()
    



    def plot_beta_2d(data_structure,preference_profile):
        X = []
        Y = []
        Z = []
        for i in range(-30,30):
            X.append(0)
            Y.append(i*1)
        for i in range(-30,30):
            X.append(i*1)
            Y.append(0)
        for i in range(-30,30):
            X.append(i*1)
            Y.append(i*1)
        for i in range(-30,30):
            X.append(i*1)
            Y.append(i*-1)

        for i in range(len(X)):
            Z.append(algorithm.cll(data_structure,preference_profile,[X[i],Y[i]],0.37))
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X,Y,Z,depthshade = 0.5) 
        # naming the x axis 
        #plt.xlabel('x - axis')
        # naming the y axis 
        #plt.ylabel('y - axis') 
        #plt.zlabel('z - axis') 
        # function to show the plot 
        plt.show()
    def plot_p_reweighting(phi):
        """
        beta = phi[1]
        alpha = phi[0]
        X = []
        Y= []
        for i in range(0,100):
            X.append(i*0.01)
            p = i*0.01
            Y.append(exp(-beta*(-np.log(p))**alpha))
        plt.plot(X,Y)
        plt.show()
            

        """
        gamma = phi[0]
        X = []
        Y= []
        for i in range(0,100):
            X.append(i*0.01)
            p = i*0.01
            Y.append((p**gamma)/(((p**gamma)+((1-p)**gamma))**(1/gamma)))
        plt.plot(X,Y)
        plt.show()
        






if __name__ == "__main__":
        plot.plot_p_reweighting([0.5,0.5])
        person0 = person(0,[1,2], 0.90,0.1)
        person1 = person(1,[3,4], 0.6,0.2)
        person2 = person(2,[2,1], 0.33,0.24)

        person3 = person(3,[1.5,1.6], 0.96,0.12)
        person4 = person(4,[3.2,3.4], 0.62,0.24)
        person5 = person(5,[2.5,1], 0.3,0.21)
        #person3 = person(3,[4,5], 0.45,0.36)
        person_matrix = [[person0,person1,person2],[person3,person4,person5]]
        #persons.append(person3)
        #persons.append(person2)

        ground_truth1 = ground_truth()
        ground_truth1.set_phi([0.5])
        ground_truth1.set_beta([1,2])

        ds = data_structure(person_matrix)
        dg = data_generator(ds,ground_truth1.get_beta(),ground_truth1.get_phi())

        #rankings = [[0,1],[1,0],[1,0]]
        rankings1 = []
        rankings2 = []
        for i in range(100):
           rankings1.append(dg.sample_a_ranking(0))
           rankings2.append(dg.sample_a_ranking(1))

        ds.build_counter_matrixs([rankings1,rankings2])
        print(ds.get_outcome_feature_matrixs())
        print(ds.get_outcome_probabilities_matrixs())
        print(ds.get_counter_matrixs())
        print()
        print(dg.get_means_matrix())
        print(dg.get_outcome_utility_vectors())
        

        #rankings.append([0,1])
        #rankings.append([1,0])
        #print(rankings)
        """
        print(rankings.count([0,1,2]))
        print(rankings.count([0,2,1]))
        print(rankings.count([1,2,0]))
        print(rankings.count([1,0,2]))
        print(rankings.count([2,0,1]))
        print(rankings.count([2,1,0]))
        """

        beta,phi = algorithm.algorithm1(ds,[0,0],[0.1,0.9],0.1,2)
        #print(beta,phi)
        """
        dg = data_generator(ds,beta,phi)
        
        
        counter_matrix = ds.get_counter_matrix()
        print(counter_matrix[0][1][1]/(counter_matrix[1][0][1]+counter_matrix[0][1][1]))
        print(exp(dg.log_p_j1_j2(1,0)))
        counter_matrix = ds.get_counter_matrix()
        print(counter_matrix[1][0][1]/(counter_matrix[1][0][1]+counter_matrix[0][1][1]))
        
        print(dg.get_outcome_utilities())
        print(ds.get_outcome_probabilities_matrix())
        print(dg.get_means())
        """


        # plot.plot_beta_3d(ds,rankings,0.37)
        #plot.plot_beta_3d(ds,rankings,0.5)
        #plot.plot_phi_3d(ds,[1])
        #plot.plot_phi_2d(ds,rankings)
        """
        utilities = dict()
        utilities['1'] = -1
        utilities['2'] = 1
        print(R_probability(perference,utilities,0.5,0.5,0.5,0))

        """
        




