import numpy as np
import os
import matplotlib.pyplot as plt

def visualize():
    y = []
    g_losses = []
    d_losses = []
    file_path = os.path.join('./paletteGAN_outputs', 'losses.txt')
    i = 0
    with open(file_path, "r") as f:
        for line in f.readlines():
            a = line.strip().split(" ")
            g_losses.append(float(a[4]))
            d_losses.append(float(a[8]))
            y.append(i)
            i += 1

    plt.plot(y, g_losses, '-p', label='generator loss')
    plt.plot(y, d_losses, '-y', label='discriminator loss')

    plt.xlabel("n epoch")
    #plt.legend(loc='upper left')
    plt.title("PaletteGAN Losses")
    plt.legend(loc="upper left")
    # save image
    #plt.savefig(title+".png")  # should before show method

    # show
    plt.show()

    return g_losses, d_losses


