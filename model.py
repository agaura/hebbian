import numpy as np

def filter(sum, param=0):
    exponential = np.exp(-sum + param)
    return 1 / (1 + exponential)

def spikes(LGN_activations, L4_weights, biases, total, penalty, exist):
    overflow = False
    vec = L4_weights @ LGN_activations + 100000 * exist * (biases - np.average(biases)) # the 100000 scales it so that the max differences normally end up at about 100

    if np.amax(vec) > 700: # overflow error happens around 709
        overflow = True
        vec = L4_weights/penalty @ LGN_activations + 100000 * exist * (biases - np.average(biases))
    
    exponent = vec + np.log(total) - np.log(np.sum(np.exp(vec)))
    
    return np.exp(exponent), overflow

class Hebbian(object):

    def __init__(self, cats, input_size, LGN_size, L4_size, param, alpha, theta, biases, exist):

        self.input_size = input_size
        self.LGN_size = LGN_size
        self.L4_size = L4_size

        self.LGN_weights = np.random.normal(size=(LGN_size, input_size))
        self.L4_weights = np.random.normal(size=(L4_size, LGN_size))
        self.L4_biases = biases
        self.L4_ideal_biases = biases

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
            prob_i = filter(self.LGN_weights[i,:] @ data, self.param)
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
        scaling = 100

        L4_input, L4_activation = self.activate(data)
        corr = np.reshape(L4_activation, (self.L4_size, 1)) @ np.reshape(L4_input, (1, self.LGN_size))

        self.L4_weights += self.alpha * (corr - np.multiply(corr, filter(self.L4_weights/scaling)))
        un_corr = np.absolute(corr-1)
        self.L4_weights += neg_lr * np.multiply(un_corr, (corr - filter(self.L4_weights/scaling)))

        self.L4_biases += 0.000005 * (self.theta * self.L4_ideal_biases - L4_activation)
        self.L4_biases = self.L4_biases / (sum(self.L4_biases) / self.theta) # normalize
        return 
    
    def associate(self, cats, data, ids):
        digits_mat = np.zeros((cats,self.L4_size))

        for i in range(len(data)):
            _, neurons = self.activate(data[i,:])

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
                self.update(data[j,:], neg_lr = 0.001*self.alpha)
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
            prediction = self.predict(data[i,:])
            if prediction == ids[i]:
                sum += 1
        
        return sum / len(data)

    def save(self, index = ''):
        np.save('save/input_size'+index, self.input_size)
        np.save('save/LGN_size'+index, self.LGN_size)
        np.save('save/L4_size'+index, self.L4_size)
        np.save('save/LGN_weights'+index, self.LGN_weights)
        np.save('save/L4_weights'+index, self.L4_weights)
        np.save('save/L4_biases'+index, self.L4_biases)
        np.save('save/L4_ideal_biases'+index, self.L4_ideal_biases)
        np.save('save/param'+index, self.param)
        np.save('save/alpha'+index, self.alpha)
        np.save('save/theta'+index, self.theta)
        np.save('save/exist'+index, self.exist)
        np.save('save/signatures'+index, self.signatures)
        return

    def load(self, index = ''):
        self.input_size = np.load('save/input_size'+index+'.npy')
        self.LGN_size = np.load('save/LGN_size'+index+'.npy')
        self.L4_size = np.load('save/L4_size'+index+'.npy')
        self.LGN_weights = np.load('save/LGN_weights'+index+'.npy')
        self.L4_weights = np.load('save/L4_weights'+index+'.npy')
        self.L4_biases = np.load('save/L4_biases'+index+'.npy')
        self.L4_ideal_biases = np.load('save/L4_ideal_biases'+index+'.npy')
        self.param = np.load('save/param'+index+'.npy')
        self.alpha = np.load('save/alpha'+index+'.npy')
        self.theta = np.load('save/theta'+index+'.npy')
        self.exist = np.load('save/exist'+index+'.npy')
        self.signatures = np.load('save/signatures'+index+'.npy')

    def ideals(self, loss_threshold, lr, cat, ideals = None):
        new_ideals = np.zeros(np.shape(ideals))
        loss = float("inf")

        new_ideals = ideals

        while loss > loss_threshold:

            L4_input = np.zeros(self.LGN_size)
            for j in range(self.LGN_size):
                prob_i = filter(self.LGN_weights[j,:] @ new_ideals, self.param)
                L4_input[j] = prob_i

            output, _ = spikes(L4_input, self.L4_weights, self.L4_biases, self.theta, 1.1, True)
            loss = np.sum(np.power(self.signatures[cat,:]-output,2)) / self.L4_size
            dL = np.reshape(self.signatures[cat,:] - loss,(1,self.L4_size))
            vec = self.L4_weights @ L4_input + 100000 * (self.L4_biases - np.average(self.L4_biases))
            dnum = np.zeros((self.L4_size, self.LGN_size))
            for j in range(self.L4_size):
                row = self.L4_weights[j,:] @ L4_input + 100000 * (self.L4_biases[j] - np.average(self.L4_biases))
                dnum[j,:] = row * self.L4_weights[j,:]
            dden = np.zeros((1,self.LGN_size))
            for j in range(self.LGN_size):
                dden[0,j] = np.sum(self.L4_weights[:,j]*np.exp(vec))
            num = np.reshape(np.exp(vec),(self.L4_size,1))
            den = np.sum(np.exp(vec))
            dL4 = (den*dnum - num@dden) / den**2
            
            sigmoid = filter(self.LGN_weights @ new_ideals, self.param)
            dLGN = np.zeros((self.LGN_size,self.input_size))
            for j in range(self.input_size):
                dLGN[:,j] = sigmoid * (np.ones((self.LGN_size))-sigmoid) * self.LGN_weights[:,j]

            din = np.reshape(dL @ dL4 @ dLGN,self.input_size)
            new_ideals -= lr*din
            new_ideals = np.clip(new_ideals,0,1)
        
        print("sum")
        print(sum(ideals-new_ideals))
        return new_ideals