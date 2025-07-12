import matplotlib.pyplot as plt

def plot_solution(t, x, exact = None, labels = None, title = None):
    plt.plot(t, x[:,], label=labels[0] if labels else "Numerical")
    if exact is not None:
        plt.plot(t, exact, '--', label=labels[1] if labels else "Exact")
    if title:
        plt.title(title)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()