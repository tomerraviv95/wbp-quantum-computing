import torch

from python_code.decoders.bp_nn import InputLayer, OddLayer, EvenLayer, OutputLayer
from python_code.decoders.decoder_trainer import DecoderTrainer
from python_code.utils.constants import CLIPPING_VAL, Phase


class BPDecoder(DecoderTrainer):
    def __init__(self):
        super().__init__()
        self.initialize_layers()
        self.total_runs = 1

    def __str__(self):
        return 'BP'

    def initialize_layers(self):
        self.input_layer = InputLayer(input_output_layer_size=self._code_bits, neurons=self.neurons,
                                      code_pcm=self.code_pcm, clip_tanh=CLIPPING_VAL,
                                      bits_num=self._code_bits)
        self.odd_layer = OddLayer(clip_tanh=CLIPPING_VAL,
                                  input_output_layer_size=self._code_bits,
                                  neurons=self.neurons,
                                  code_pcm=self.code_pcm)
        self.even_layer = EvenLayer(CLIPPING_VAL, self.neurons, self.code_pcm)
        self.output_layer = OutputLayer(neurons=self.neurons,
                                        input_output_layer_size=self._code_bits,
                                        code_pcm=self.code_pcm)

    def calc_loss(self, decision, labels):
        pass

    def forward(self, x, mode: Phase = Phase.TEST):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        x = x.float()
        # initialize parameters
        output_list = [0] * self.iteration_num

        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        output_list[0] = x + self.output_layer.forward(even_output, mask_only=self.output_layer)

        # now start iterating through all hidden layers i>2 (iteration 2 - Imax)
        for i in range(0, self.iteration_num - 1):
            # odd - variables to check
            odd_output = self.odd_layer.forward(even_output, x, llr_mask_only=self.odd_llr_mask_only)
            # even - check to variables
            even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
            # outputs layer
            output = x + self.output_layer.forward(even_output, mask_only=self.output_mask_only)
            output_list[i + 1] = output.clone()

        if mode == Phase.TEST:
            decoded_words = torch.round(torch.sigmoid(-output_list[-1]))
            return decoded_words
        return output_list
