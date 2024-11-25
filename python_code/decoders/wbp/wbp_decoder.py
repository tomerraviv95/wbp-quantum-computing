import os

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import RMSprop, Adam

from dir_definitions import BP_WEIGHTS
from python_code import DEVICE
from python_code.datasets.channel_dataset import ChannelModelDataset
from python_code.decoders.bp_nn import InputLayer, OddLayer, EvenLayer, OutputLayer
from python_code.decoders.decoder_trainer import DecoderTrainer, LR, EPOCHS
from python_code.utils.constants import Phase, CLIPPING_VAL


class WBPDecoder(DecoderTrainer):
    def __init__(self):
        super().__init__()
        self.initialize_layers()
        self.train_channel_dataset = ChannelModelDataset()
        self.load_or_train_model()

    # setup the optimization algorithm
    def deep_learning_setup(self, lr: float):
        """
        Sets up the optimizer and loss criterion
        """

        # self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                              weight_decay=0.0005, betas=(0.5, 0.999))
        self.criterion = BCEWithLogitsLoss().to(DEVICE)

    def load_or_train_model(self):
        if not os.path.exists(BP_WEIGHTS):
            os.makedirs(BP_WEIGHTS)
        self._model_name = f'WBP_{self.iteration_num}_' + str(self.train_channel_dataset)
        model_path = os.path.join(BP_WEIGHTS, self._model_name)
        # Check if the model already exists on disk
        if os.path.exists(model_path):
            # Load the existing model
            self.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {self._model_name}")
        else:
            # Save the model to disk
            self.train()
            torch.save(self.state_dict(), model_path)
            print(f"Model saved to {self._model_name}")

    def __str__(self):
        return 'F-WBP'

    def initialize_layers(self):
        self.input_layer = InputLayer(input_output_layer_size=self._code_bits, neurons=self.neurons,
                                      code_pcm=self.code_pcm, clip_tanh=CLIPPING_VAL,
                                      bits_num=self._code_bits)
        self.even_layer = EvenLayer(CLIPPING_VAL, self.neurons, self.code_pcm)
        self.odd_layer = OddLayer(clip_tanh=CLIPPING_VAL,
                                  input_output_layer_size=self._code_bits,
                                  neurons=self.neurons,
                                  code_pcm=self.code_pcm)
        self.output_layer = OutputLayer(neurons=self.neurons,
                                        input_output_layer_size=self._code_bits,
                                        code_pcm=self.code_pcm)

    def calc_loss(self, decision, labels):
        loss = self.criterion(input=-decision[-1], target=labels)
        if self.multi_loss_flag:
            for iteration in range(self.iteration_num - 1):
                loss += self.criterion(input=-decision[iteration], target=labels)
        return loss

    def train(self):
        self.deep_learning_setup(LR)
        for _ in range(EPOCHS):
            cx, tx, rx = self.train_channel_dataset.__getitem__()
            output_list = self.forward(rx, mode=Phase.TRAIN)
            # calculate loss
            loss = self.calc_loss(decision=output_list[-self.iteration_num:], labels=cx)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self, x, mode=Phase.TEST):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        x = x.float()
        # initialize parameters
        output_list = [0] * self.iteration_num
        output_list[-1] = torch.zeros_like(x)

        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        output_list[0] = x + self.output_layer.forward(even_output,
                                                       mask_only=self.multiloss_output_mask_only)

        # now start iterating through all hidden layers i>2 (iteration 2 - Imax)
        for i in range(self.iteration_num - 1):
            odd_output = self.odd_layer.forward(even_output, x, llr_mask_only=self.odd_llr_mask_only)
            # even - check to variables
            even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
            # output layer
            output = x + self.output_layer.forward(even_output, mask_only=self.multiloss_output_mask_only)
            output_list[i + 1] = output.clone()
        output_list[-1] = x + self.output_layer.forward(even_output, mask_only=self.output_mask_only)
        if mode == Phase.TEST:
            decoded_words = torch.round(torch.sigmoid(-output_list[-1]))
            return decoded_words
        return output_list
