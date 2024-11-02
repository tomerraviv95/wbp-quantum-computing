import matplotlib as mpl
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


def plot_ber_vs_flip_prob(evaluator, flip_probs):
    """
    Plots BER vs. Flipping Probability for a range of flipping probabilities.

    Parameters:
        evaluator: Object with an evaluate() method that calculates BER at a specified flipping probability.
        flip_probs (list): List of flipping probabilities to evaluate.
    """
    ber_values = []

    # Evaluate BER over each flipping probability
    for prob in flip_probs:
        conf.set_value('p', prob)  # Set the flipping probability in the config
        ber = evaluator.evaluate()
        ber_values.append(ber)
        print(f"Flipping Probability: {prob}, BER: {ber}")

    # Plot BER vs. Flipping Probability
    plt.figure(figsize=(9.5, 6.45))
    plt.plot(flip_probs, ber_values, marker='o', linestyle='-', color='b', label=str(evaluator.decoder))
    plt.yscale('log')  # Log scale for BER
    plt.xlabel("Bit Flipping Probability")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"{conf.code_type} ({conf.code_bits},{conf.message_bits}) - {conf.channel_model}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(f'ber_vs_flip_prob_{conf.channel_model}_{conf.code_type}_{conf.code_bits}_{conf.message_bits}.png')
    plt.show()


if __name__ == "__main__":
    evaluator = Evaluator()
    flip_probs = [0.01, 0.05, 0.1, 0.15, 0.2]  # Example flipping prob
    plot_ber_vs_flip_prob(evaluator, flip_probs)
