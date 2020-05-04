import numpy as np

def filter(sum, param=0):
    exponential = np.exp(-sum + param)
    return 1 / (1 + exponential)

def spikes(LGN_activations, L4_weights, biases, total, penalty, exist):
    overflow = False
    vec = L4_weights @ LGN_activations + 100000 * exist * (biases - np.average(biases))

    if np.amax(vec) > 700: # overflow error happens around 709
        overflow = True
        vec = L4_weights/penalty @ LGN_activations + 100000 * exist * (biases - np.average(biases))
    
    exponent = vec + np.log(total) - np.log(np.sum(np.exp(vec)))
    
    return np.exp(exponent), overflow

class Hebbian(object):

    def __init__(self, cats, LGN_size, L4_size, param, alpha, theta, exist):

        self.LGN_size = LGN_size
        self.L4_size = L4_size

        self.LGN_weights = np.random.normal(size=(LGN_size, cats))
        self.L4_weights = np.random.normal(size=(L4_size, LGN_size))
        self.L4_biases = np.asarray([theta*1./cats]*L4_size)
        self.L4_ideal_biases = np.asarray([theta*1./cats]*L4_size)

        self.param = param
        self.alpha = alpha
        self.theta = theta

        self.exist = exist
        self.signatures = np.zeros((cats, self.theta))

        self.L4_history = []

        return

    def activate(self, data):

        L4_input = np.zeros(self.LGN_size)

        for i in range(self.LGN_size):
            prob_i = filter(self.LGN_weights[i,data], self.param)
            if np.random.random_sample() < prob_i:
                L4_input[i] = 1

        penalty = 2
        output, overflow = spikes(L4_input, self.L4_weights, self.L4_biases, self.theta, penalty, self.exist)
        if overflow == True:
            self.L4_weights = self.L4_weights/penalty
        order = np.argsort(output)

        index = 0
        last_index = 0
        new_order = np.zeros(np.size(order), dtype=int)
        while index < np.size(order):
            if last_index + 1 < np.size(order):
                while output[order[last_index+1]] == output[order[index]]:
                    last_index += 1
                    if last_index+1 >= np.size(order): break
            length = last_index - index + 1
            shuffle = np.zeros(length, dtype=int)
            for i in range(length):
                shuffle[i] = index + i
            np.random.shuffle(shuffle)
            for i in range(length):
                new_order[index+i] = order[shuffle[i]]
            index = last_index + 1
            last_index = index
        order = np.asarray(new_order)

        L4_activation = np.zeros(self.L4_size)
        for i in range(self.theta):
            L4_activation[order[-i-1]] = 1
        
        return L4_input, L4_activation

    def update(self, data, neg_lr = 0):
        scaling = 10000

        L4_input, L4_activation = self.activate(data)
        corr = np.reshape(L4_activation, (self.L4_size, 1)) @ np.reshape(L4_input, (1, self.LGN_size))
        neg_corr = corr - 1
        self.L4_weights = self.L4_weights + self.alpha * corr + neg_lr * neg_corr
        self.L4_weights += self.alpha * (corr - np.multiply(corr, filter(self.L4_weights/scaling)))
        un_corr = np.absolute(corr-1)
        self.L4_weights += neg_lr * np.multiply(un_corr, (corr - filter(self.L4_weights/scaling)))
        self.L4_biases += self.alpha/10000 * (self.theta * self.L4_ideal_biases - L4_activation)
        self.L4_biases = self.L4_biases / (sum(self.L4_biases) / self.theta) # normalize
        return 

    def associate(self, cats, data, ids):
        digits_mat = np.zeros((cats,self.L4_size))

        for i in range(len(data)):
            _, neurons = self.activate(data[i])

            truth = ids[i]
            for j in range(self.L4_size):
                if neurons[j] == 1:
                    digits_mat[truth,j] += 1
                
        averages = np.zeros((cats,self.L4_size))
        for i in range(cats):
            if np.sum(digits_mat[i,:]) != 0:
                averages[i,:] = self.theta * digits_mat[i,:] / np.sum(digits_mat[i,:])

        self.signatures = averages
        print(averages)
        return 

    def train(self, data, epochs, ids):
        for i in range(epochs):
            for j in range(len(data)):
                if j % 100 == 0:
                    print(j)
                self.update(data[j], neg_lr = 0.001*self.alpha)
        self.associate(len(self.signatures), data, ids)
        return
    
    def predict(self, data):
        _, neurons = self.activate(data)

        distance = np.zeros(len(self.signatures))
        for i in range(len(self.signatures)):
            distance[i] = np.sum((neurons-self.signatures[i,:])**2)

        return np.argmin(distance)

    def test(self, data, ids):
        sum = 0
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if prediction == ids[i]:
                sum += 1
        
        return sum / len(data)