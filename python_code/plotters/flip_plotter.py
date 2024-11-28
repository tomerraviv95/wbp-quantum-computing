import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

from python_code import conf
from python_code.evaluator import Evaluator

# Custom matplotlib configurations
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

METHODS_TO_COLORS = {'hd': 'red', 'hard_bp': 'green', 'wbp': 'blue'}
METHODS_TO_MARKERS = {'hd': 'p', 'hard_bp': 'o', 'wbp': 's'}
METHODS_TO_LINESTYLES = {'hd': '-.', 'hard_bp': '-', 'wbp': '--'}


def plot_ber_vs_flip_prob(methods):
    """
    Plots BER vs. Flipping Probability for a range of flipping probabilities.

    Parameters:
        evaluator: Object with an evaluate() method that calculates BER at a specified flipping probability.
        flip_probs (list): List of flipping probabilities to evaluate.
    """
    conf.set_value('channel_model', 'BSC')
    flip_probs = [0.0001, 0.001, 0.005, 0.01, 0.025]  # Example flipping prob
    # Plot BER vs. SNR
    plt.figure(figsize=(8, 6))
    for method in methods:
        print(f'Decoding Method: {method}')
        conf.set_value('decoder_type', method)  # Set the method value in the config
        ber_values = []
        # Evaluate BER over each SNR value
        for prob in flip_probs:
            conf.set_value('p', prob)  # Set the SNR value in the config
            evaluator = Evaluator()
            ber = evaluator.evaluate()
            ber_values.append(ber)
            print(f"Prob: {prob}, BER: {ber}")
        plt.plot(flip_probs, ber_values, marker=METHODS_TO_MARKERS[method], linestyle=METHODS_TO_LINESTYLES[method],
                 color=METHODS_TO_COLORS[method], label=str(evaluator.decoder))
    plt.yscale('log')  # Log scale for BER
    plt.xlabel("Bit Flipping Probability")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"{conf.code_type} ({conf.code_bits},{conf.message_bits}) - {conf.channel_model}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(f'ber_vs_flip_prob_{conf.channel_model}_{conf.code_type}_{conf.code_bits}_{conf.message_bits}.png')
    plt.show()


if __name__ == "__main__":
    methods = ['hd', 'hard_bp']
    plot_ber_vs_flip_prob(methods)
