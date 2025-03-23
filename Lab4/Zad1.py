import numpy as np

class NeuralNetwork:
    def __init__(self, weights1, bias1, weights2, bias2):
       
        self.weights1 = np.array(weights1)  
        self.bias1 = np.array(bias1)        
        self.weights2 = np.array(weights2).reshape(-1, 1)  
        self.bias2 = np.array(bias2)        

    def sigmoid(self, z):
        
        return 1 / (1 + np.exp(-z))

    def forward(self, inputs):

        results = []
        
        for input in inputs:
            hidden1 = self.bias1[0]
            hidden2 = self.bias1[1]

            for i in range (len(input)):
                hidden1 += input[i] * self.weights1[i][0] 
                hidden2 += input[i] * self.weights1[i][1] 
        
            hidden1act = self.sigmoid(hidden1) 
            hidden2act = self.sigmoid(hidden2)

            output = (hidden1act * self.weights2[0][0]) + (hidden2act * self.weights2[1][0]) + self.bias2[0]
            

            results.append(output)
        return np.array(results)


if __name__ == "__main__":
    
    weights1 = [
    [-0.46, 0.78],  
    [0.97, 2.1],    
    [-0.39, -0.58]  
]
    bias1 = [0.8, 0.44]
    weights2 = [
    [-0.81],  
    [1.03]    
]
    bias2 = [-0.23]

    
    nn = NeuralNetwork(weights1, bias1, weights2, bias2)

    
    inputs = np.array([
        [23, 75, 176],  # Person 1
        [25, 67, 180],  # Person 2
        [28, 120, 175],  # Person 3
        [22, 65, 165],  # Person 4
        [46, 70, 187],  # Person 5
        [50, 68, 180],  # Person 6
        [48, 97, 178],  # Person 7
    ])

    
    
    predictions = nn.forward(inputs)
    print("Predictions (Probability of playing volleyball):")
    for p in predictions:
        print(p)

    
    for i, pred in enumerate(predictions):
        if pred > 0.5:
            print(f"Person {i+1} will play volleyball.")
        else:
            print(f"Person {i+1} will not play volleyball.")