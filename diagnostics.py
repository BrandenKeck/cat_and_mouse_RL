import numpy as np

class network_diagnostics():

    # Initialize network tracking properties
    def __init__(self):

        # MSE Tracking
        self.track_mse = True
        self.mse = []
        self.mse_queue_length = 500000

        # Network Value Tracking
        self.track_layer_gradients = True
        self.layer_gradients = None
        self.lg_queue_length = 500000

    # Run Diagnostics Functions
    def run_diag(self, Y_hat, Y):

        if self.track_mse:
            self.mse_calc(Y_hat, Y)

    # MSE Calculation
    def mse_calc(self, Y_hat, Y):
        # Calculate Mean Squared Error
        sqr_err = (Y_hat - Y) ** 2
        sum_sqr_err = np.sum(sqr_err)
        self.mse.append(sum_sqr_err / len(Y))
        while len(self.mse) > self.mse_queue_length:
            self.mse.pop(0)

class grid_diagnostics():

    # Initialize network tracking properties
    def __init__(self):
        pass