## ==== BEGIN EVALUATION PORTION

class RNNEncoderDecoderLMWithAttention(torch.nn.Module):
    """ Implements an Encoder-Decoder network, using RNN units, augmented with attention. """

    # Feel free to add additional parameters to __init__
    def __init__(self,src_vocab_size, tgt_vocab_size, embd_dims, hidden_size, num_layers=1, dropout=0.1):
        """ Initializes the encoder-decoder network, implemented via RNNs.

        Args:
            src_vocab_size (int): Source vocabulary size.
            tgt_vocab_size (int): Target vocabulary size.
            embd_dims (int): Embedding dimensions.
            hidden_size (int): Size/Dimensions for the hidden states.
        """

        super(RNNEncoderDecoderLMWithAttention, self).__init__()

        # Dummy parameter to track the model device. Do not modify.
        self._dummy_param = torch.nn.Parameter(torch.Tensor(0), requires_grad=False)

        # BEGIN CODE : enc-dec-rnn-attn.init

        # ADD YOUR CODE HERE

        self.rnn_attention = True

        self.encoder_embedding = torch.nn.Embedding(src_vocab_size, embd_dims)
        self.decoder_embedding = torch.nn.Embedding(tgt_vocab_size, embd_dims)

        self.dropout = torch.nn.Dropout(dropout)

        dropout = dropout if num_layers > 1 else 0
        self.encoder_rnn = torch.nn.GRU(embd_dims, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.decoder_rnn = torch.nn.GRU(embd_dims+2*hidden_size, 2*hidden_size, num_layers=num_layers, batch_first=True, dropout = dropout)

        self.decoder_output = torch.nn.Linear(hidden_size, tgt_vocab_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.attention = AttentionModule(embd_dims, hidden_size, hidden_size, hidden_size, hidden_size, dropout)


        # END CODE

    @property
    def device(self):
        return self._dummy_param.device

    def log_probability(self, seq_x, seq_y):
        """ Compute the conditional log probability of seq_y given seq_x, i.e., log P(seq_y | seq_x).

        Args:
            seq_x (torch.tensor): Input sequence of tokens, of shape [src_seq_len] (no batch dim)
            seq_y (torch.tensor): Output sequence of tokens, of shape [tgt_seq_len] (no batch dim)

        Returns:
            float: Log probability of generating sequence y, given sequence x.
        """

        # BEGIN CODE : enc-dec-rnn-attn.probability

        # ADD YOUR CODE HERE

        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Convert seq_x to batch format: [1, src_seq_len]
            seq_x = seq_x.unsqueeze(0).to(self.device)
            seq_y = seq_y.unsqueeze(0).to(self.device)

            # Encode the input sequence
            encoder_embedded = self.encoder_embedding(seq_x)  # [1, src_seq_len, embd_dims]
            encoder_output, encoder_hidden = self.encoder_rnn(encoder_embedded)

            # Initialize decoder: start with BOS token (assumed index 1, as in attentions)
            decoder_input_token = torch.tensor([[1]], dtype=torch.long, device=self.device)
            decoder_hidden_state = encoder_hidden

            total_log_prob = 0.0

            # Teacher forcing: iterate over each token in seq_y (assumed to be a 1D tensor)
            for target_token in seq_y[0, :]:
                # Embed the current decoder input token
                decoder_embedded = self.decoder_embedding(decoder_input_token)  # [1, 1, embd_dims]

                # Compute attention using the last layer of the current decoder hidden state
                attnention_weights = self.attention(encoder_output, decoder_hidden_state[-1])  # [1, 1, src_seq_len]
                context_vector = torch.bmm(attnention_weights, encoder_output)  # [1, 1, hidden_size]

                # Concatenate the embedded token and context vector
                decoder_input = torch.cat([decoder_embedded, context_vector], dim=2)  # [1, 1, embd_dims + hidden_size]

                # Decode step with current hidden state
                decoder_output, decoder_hidden_state = self.decoder_rnn(decoder_input, decoder_hidden_state)
                decoder_logits = self.decoder_output(decoder_output)  # [1, 1, tgt_vocab_size]
                decoder_log_probs = torch.nn.functional.log_softmax(decoder_logits, dim=-1)

                # Accumulate the log probability of the correct target token
                token_log_prob = decoder_log_probs[0, 0, target_token.item()]
                total_log_prob += token_log_prob.item()

                # Use teacher forcing: set next input to the current target token
                decoder_input_token = target_token.unsqueeze(0).unsqueeze(0)

            return total_log_prob
        # END CODE

    def attentions(self, seq_x, terminate_token, max_length):
        """ Obtain attention over a sequence for decoding to the target language.

        Args:
            seq_x (torch.tensor): Tensor representing the source sequence, of shape [src_seq_len] (no batch dim)
            terminate_token (int): Token to use as EOS, to stop generating outputs.
            max_length (int): Maximum length to use to terminate the sampling.

        Returns:
            tuple[torch.tensor, torch.tensor]:
                A tuple of two tensors: the attentions over individual output tokens ([tgt_seq_len, src_seq_len])
                and the best output tokens ([tgt_seq_len]) per sequence step, based on greedy sampling.
        """

        # BEGIN CODE : rnn-enc-dec-attn.attentions

        # ADD YOUR CODE HERE

        self.eval()  # Set model to evaluation mode

        with torch.no_grad():
            # Convert seq_x to batch format: [1, src_seq_len]
            seq_x = seq_x.unsqueeze(0).to(self.device)

            # Encode input sequence
            encoder_embedded = self.encoder_embedding(seq_x)  # [1, src_seq_len, embd_dims]
            encoder_output, encoder_hidden = self.encoder_rnn(encoder_embedded)

            # Start decoding with a BOS token (assuming index 0 as BOS; adjust if needed)
            seq_y = torch.tensor([[1]], dtype=torch.long, device=self.device)
            # Initialize decoder hidden state with encoder's final hidden state
            decoder_hidden_state = encoder_hidden

            attentions = []
            decoded_tokens = []

            for _ in range(max_length):
                # Embed the current decoder token
                decoder_embedded = self.decoder_embedding(seq_y[:, -1]).unsqueeze(1)  # [1, 1, embd_dims]

                # Compute attention using the last layer of the current decoder hidden state (consistent with forward)
                attnention_weights = self.attention(encoder_output, decoder_hidden_state[-1])  # [1, 1, src_seq_len]
                # Compute context vector
                context_vector = torch.bmm(attnention_weights, encoder_output)  # [1, 1, hidden_size]

                # Concatenate the embedded token and context vector
                decoder_input = torch.cat([decoder_embedded, context_vector], dim=2)  # [1, 1, embd_dims + hidden_size]

                # Decode step with current hidden state
                decoder_output, decoder_hidden_state = self.decoder_rnn(decoder_input, decoder_hidden_state)
                decoder_logits = self.decoder_output(decoder_output)  # [1, 1, tgt_vocab_size]
                decoder_log_probs = torch.nn.functional.log_softmax(decoder_logits, dim=-1)

                # Greedy decoding: select the token with highest log probability
                next_token = decoder_log_probs.argmax(dim=-1)  # [1, 1]
                decoded_tokens.append(next_token.item())

                # Store the attention weights (squeeze the batch dim)
                attentions.append(attnention_weights.squeeze(0).cpu())

                # Stop if the terminate token is generated
                if next_token.item() == terminate_token:
                    break

                # Append the predicted token to the decoder input sequence for the next step
                seq_y = torch.cat([seq_y, next_token], dim=1)

            # Stack attention weights to form a tensor of shape [tgt_seq_len, src_seq_len]
            attentions = torch.stack(attentions, dim=0)
        return attentions.squeeze(), torch.tensor(decoded_tokens)


        # END CODE

    def forward(self, inputs, decoder_inputs=None, decoder_hidden_state=None, output_attention=False):
        """ Performs a forward pass over the encoder-decoder network.

            Accepts inputs for the encoder, inputs for the decoder, and hidden state for
                the decoder to continue generation after the given input.

        Args:
            inputs (torch.Tensor): tensor of shape [batch_size?, src_seq_len]
            decoder_inputs (torch.Tensor): Decoder inputs, as tensor of shape [batch_size?, 1]
            decoder_hidden_state (any): tensor to represent decoder hidden state from time step T-1.
            output_attention (bool): If true, this function should also return the
                associated attention weights for the time step, of shape [batch_size?, 1, src_seq_len].

        Returns:
            tuple[torch.Tensor, any]: output from the decoder, and associated hidden state for the next step.

            Decoder outputs should be log probabilities over the target vocabulary.

        Example:
        >>> model = RNNEncoderDecoderWithAttention(*args, **kwargs)
        >>> output, hidden = model(..., output_attention=False)
        >>> output, hidden, attn_weights = model(..., output_attention=True)
        """

        # BEGIN CODE : enc-dec-rnn-attn.forward

        # ADD YOUR CODE HERE

        # Encode the inputs
        encoder_embedded = self.dropout(self.encoder_embedding(inputs))          # [batch, src_seq_len, embd_dims]

        encoder_output, encoder_hidden = self.encoder_rnn(encoder_embedded)  # encoder_hidden: [num_layers, batch, hidden_size]

        # If no decoder hidden state is provided, use the full encoder hidden state.
        if decoder_hidden_state is None:
            decoder_hidden_state = encoder_hidden.view(self.num_layers, 2, -1, self.hidden_size).sum(dim=1).unsqueeze(0)

        
        decoder_embedded = self.dropout(self.decoder_embedding(decoder_inputs))

        attnention_weights = self.attention(encoder_output, decoder_hidden_state)

        context_vector = torch.bmm(attnention_weights, encoder_output)  # [batch, 1, hidden_size]

        decoder_input = torch.cat([decoder_embedded, context_vector], dim=2)  # [batch, 1, embd_dims + hidden_size]

        print("Decoder input shape: ", decoder_input.shape)
        print("Decoder Hidden State shape: ", decoder_hidden_state.shape)
        decoder_output, decoder_hidden_state = self.decoder_rnn(decoder_input, decoder_hidden_state)

        decoder_logits = self.dropout(self.decoder_output(decoder_output))  # [batch, 1, tgt_vocab_size]

        decoder_log_probs = torch.nn.functional.log_softmax(decoder_logits, dim=-1)

        if output_attention:
            return decoder_log_probs, decoder_hidden_state, attnention_weights

        return decoder_log_probs, decoder_hidden_state

        # END CODE

## ==== END EVALUATION PORTION






## ==== BEGIN EVALUATION PORTION

class AttentionModule(torch.nn.Module):
    """ Implements an attention module """

    # Feel free to add additional parameters to __init__
    def __init__(self, input_size, encoder_dim, decoder_dim, attention_dim, hidden_size, dropout=0.1):
        """ Initializes the attention module.
            Feel free to declare any parameters as required. """

        super(AttentionModule, self).__init__()

        # BEGIN CODE : attn.init

        # ADD YOUR CODE HERE

        self.Wa = torch.nn.Linear(2*hidden_size, 2*hidden_size)
        self.scale_factor = torch.nn.Parameter(torch.tensor(1.0))
        # END CODE

    def forward(self, encoder_outputs, decoder_hidden_state):
        """ Performs a forward pass over the module, computing attention scores for inputs.

        Args:
            encoder_outputs (torch.Tensor): Output representations from the encoder, of shape [batch_size?, src_seq_len, output_dim].
            decoder_hidden_state (torch.Tensor): Hidden state from the decoder at current time step, of appropriate shape as per RNN unit (with optional batch dim).

        Returns:
            torch.Tensor: Attentions scores for given inputs, of shape [batch_size?, 1, src_seq_len]
        """

        # BEGIN CODE : attn.forward

        # ADD YOUR CODE HERE

        # Transform decoder state for better alignment
        decoder_hidden_state = decoder_hidden_state.squeeze(1)  # Remove time step dimension
        decoder_hidden_state = self.Wa(decoder_hidden_state)  # [batch_size, hidden_dim]
        decoder_hidden_state = decoder_hidden_state.unsqueeze(2)  # [batch_size, hidden_dim, 1]

        # Compute raw scores (scaled dot-product)
        attn_scores = torch.bmm(encoder_outputs, decoder_hidden_state).squeeze(2)  # [batch_size, src_seq_len]

        # Apply scaling (√d_k) and learnable weight
        attn_scores = attn_scores / (encoder_outputs.size(-1) ** 0.5)
        attn_scores = self.scale_factor * attn_scores

        # Softmax over source sequence length
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1).unsqueeze(1)  # [batch_size, 1, src_seq_len]

        return attn_weights




        # END CODE

## ==== END EVALUATION PORTION