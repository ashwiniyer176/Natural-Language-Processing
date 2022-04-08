from sklearn.model_selection import train_test_split
'''
This is an sklearn trainer that reduces the need to write boilerplate code over and over again
'''


class Trainer:
    def __init__(self, model, X, y, metrics):
        """
        model: sklearn model instance
        X: input for the given model
        y: target values for the given model
        metrics: dictionary of the form {'metric name':metric_callable}. Sklearn metrics supported
        """
        self.model = model
        self.X = X
        self.y = y
        self.metrics = metrics

    def _random_split(self, test_size):
        """
        Args:
        test_size(float): float value between 0 and 1 indicating the ratio of the split.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42)

    def _sequential_split(self, test_size):
        """
        Args:
        test_size(float): float value between 0 and 1 indicating the ratio of the split.
        """
        self.X_train = self.X.iloc[:int(test_size*len(self.X)), :]
        self.y_train = self.y.iloc[:int(test_size*len(self.X)), :]
        self.X_test = self.X.iloc[int(test_size*len(self.X)):, :]
        self.y_test = self.y.iloc[int(test_size*len(self.X)):, :]

    def split_data(self, type="random", test_size=0.25):
        """
        Description: Splits data according to type specified
        Args:

        type(str): {'random','sequential'} default='random'.
        Splits the data randomly or in sequence.  

        test_size(float): float value between 0 and 1 indicating the ratio of the split.
        default=0,25
        """
        if type == "random":
            self._random_split(test_size)
        elif type == "sequential":
            self._sequential_split(test_size)

    def train_and_evaluate(self):
        """
        Trains and evaluates the model's performance on the validation data created internally
        """
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        self._evaluate(predictions)

    def _evaluate(self, predictions, y_true=None):
        """
        Args:
        predictions(numpy array): The predicted values
        """
        for metric in self.metrics:
            if y_true == None:
                print(
                    f"{metric}:{self.metrics[metric](self.y_test,predictions)}")
            else:
                print(f"{metric}:{self.metrics[metric](y_true,predictions)}")
