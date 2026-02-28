import matplotlib.pyplot as plt

def plot_reconstruction(t, y, yhat, washout):
    plt.figure(figsize=(10,4))
    plt.plot(t[washout:], y[washout:], "--", label="true")
    plt.plot(t[washout:], yhat[washout:], label="reconstructed")
    plt.legend()
    plt.show()