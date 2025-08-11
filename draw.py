import torch
import matplotlib.pyplot as plt

ctd_poison = torch.load('./cache/ctd_poisoned.pt')
ctd_clean = torch.load('./cache/ctd_clean.pt')

def plot_ctd():
    # Do a histogram of the CTD values
    plt.figure(figsize=(10, 5))
    plt.hist(ctd_poison, bins=50, alpha=0.5, label='Poisoned Data', color='red')
    plt.hist(ctd_clean, bins=50, alpha=0.5, label='Clean Data', color='blue')
    plt.title('CTD Poison vs Clean Data')
    plt.xlabel('CTD Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == "__main__":
    plot_ctd()