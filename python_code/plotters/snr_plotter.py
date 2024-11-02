import matplotlib as mpl
import matplotlib.pyplot as plt

from python_code import conf
from python_code.evaluator import Evaluator

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


def snr_plotter(methods):
    """
    Plots BER vs. SNR for a range of SNR values.

    Parameters:
        evaluator: Object with an evaluate() method that calculates BER at a specified SNR.
        snr_list (list): List of SNR values to evaluate.
    """

    # Plot BER vs. SNR
    plt.figure(figsize=(8, 6))
    snr_list = list(range(3, 7, 1))
    for method in methods:
        print(f'Decoding Method: {method}')
        conf.set_value('decoder_type', method)  # Set the method value in the config
        ber_values = []
        # Evaluate BER over each SNR value
        for snr in snr_list:
            conf.set_value('snr', snr)  # Set the SNR value in the config
            evaluator = Evaluator()
            ber = evaluator.evaluate()
            ber_values.append(ber)
            print(f"SNR: {snr} dB, BER: {ber}")
        plt.plot(snr_list, ber_values, marker='o', linestyle='-', color='b', label=str(evaluator.decoder))
    plt.yscale('log')  # Log scale for BER
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"{conf.code_type} ({conf.code_bits},{conf.message_bits}) - {conf.channel_model}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(f'ber_vs_snr_{conf.channel_model}_{conf.code_type}_{conf.code_bits}_{conf.message_bits}.png')
    plt.show()


if __name__ == "__main__":
    methods = ['wbp']
    snr_plotter(methods)
