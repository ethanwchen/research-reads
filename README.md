# Attention Is All You Need

## Intro

This paper introduces the transformer, depending on attention mechanisms and dispensing with convolutions/recurrence entirely. This is a model that achieves 28.4 BLEU on the WMT 2014 English-to-German translation task. On English-to-French, the model establishes a score of 41.8 after 3.5 days of training on eight GPUs.

Recurrent neural networks have been established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation. 

Sequence modeling is the process of learning patterns and structures in sequential data where order matters. An example would be a one-to-one task where input would map to a single output (image classification). Sequence transduction is a specialized sequence modeling task where an input sequence is transformed into some output sequence of different lengths. For example, "Hello, how are you?" would be converted into "Bonjour, comment ca va?". This means learning about patterns from sequential data. They typically factor computation along with symbol positions of the input and output sequences. Aligning the positions to step in computation time, they generate a sequence of hidden states $h_t$ as a function of the previous hidden state $h_{t-1}$ and the input for position $t$. This is how RNN's work right now. This inherently precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Inherently sequential nature references RNNs processing a step at a time meaning you can't compute $h_t$ without $h_{t-1}$. This means you cannot compute the time stamps in parallel, which is bad because training becomes very slow as each time step has to wait for the previous one to finish. Normally in deep learning, we use batching to speed about processing multiple examples but RNNs cannot efficiently handle large batches. 

Recently, there has been factorization tricks to make these RNN based models faster. For example, instead of processing to every word in a sentence, the model might spend more effort on key words and ignore the more trivial ones. It makes models perform better as well because there is focus on computation in the most important parts of the input. However, they are still fundamentally sequential and you can't parallelize computations the way you can in Transformers. 

Attention mechanisms are now an integral part of sequence modeling and transduction models, as models can focus on important parts of an input sequence dynamically. Unlike RNNs, which process data step by step, attention can directly connect any two words in a sequence. For example, "The cat, which was hungry, ate the fish" would link "cat" to "ate" directly. Before the Transformer, most models combined attention with RNNs. For example, Seq2Seq with Attention was popular before Transformers. 

This paper introduces the Transformer, which removes RNNs (sequential processing) altogether. Transformeers can, instead, use self-attention to look at allwords in a sequence at once. The model can draw global dependencies, meaning it can connect words no matter how far apart they are. Basically, all the words in a sequence can be processed at once and this makes training much faster.

## Background

Even though CNNs are an option, the computational cost grows linearly with sequence length. The farther apart two words are in a sequence, the harder it is for CNNs to relate them. This makes it more difficult to learn dependencies between distant positions. Transformers compute dependencies in constant time (O(1)), regardless of sequence length. Attention mechanisms average over multiple words, which can blur fine-grained details. We use multi-head attention instead to allow the model to focus on different parts of the sequence simultaneously. 

Self-attention is an attention mechanism referring to where each word in a sequence can directly relate to another word, allowing it to capture long-range dependencies efficiently. Unlike CNNs and RNNs, self-attention isn't distance dependent.

So far, self-attention has already been used in tasks like text summarization, reading comprehension, and textual entailment.

This paper is essentially the core contribution of the Transformer model, basically the only Transformer that uses self-attention.

Most of the better sequence transduction models have an encoder-decoder structure [5, 2, 35]. The encoder would map an input sequence of symbol representations ($x_{1}$, ... , $x_{n}$) to a sequence of continuous representations z = ($z_{1}$, ... , $z_{n}$). Given $z$, the decoder would then generate some output sequence ($y_{1}$, ... , $y_{m}$) of symbols one element at a time. At each step, the model is auto-regressive and consumes the previously generated symbol as additional input when generating the text.

## Self-Attention and Multi-Head Attention

Self-attention is a key component of the Transformer model. Unlike RNNs, which process words sequentially and carry information from previous time steps, self-attention allows the model to directly compare all words in a sequence to each other at once. This makes it highly efficient for learning dependencies between words regardless of distance.

The way it works.

Self-attention assigns a weight (attention score) to each word in the input sequence, determining how much focus each word should have on every other word. Given some input sequence of words represented as vectors, the model would compute three essential matrices for each word.

- Query: the word currently being attended to
- Key: words that the model compares to the query
- Value: contains the information to be passed along to the next layer

Attention score would then be calculated using the scaled dot-product attention formula: 

Attention(Q, K, V) = softmax($\frac{QK^{T}}{\sqrt{d_{k}}}$)V

where $QK^{T}$ computes how similar the query and key are. $\sqrt{d_{k}}$ is a scaling factor to prevent large values. The softmax function normalizes the attention scores to sum to 1.

Multi-Head Attention.

A single self-attention mechanism might fail to capture different aspects of meaning so Transformers use Mult-Head Attention, applying self-attention multiple times in parallel with different learned weight matrices. Each head attends to different parts of the sentence, capturing various linguistic features. The output of multiple attention heads is then concatenated and projected back into the model.

MultHead(Q, K, V) = Concat(head_1, head_2, head_h)W^{O}

By using multiple attention heads, the Transformer ensures a richer representation of input sequences, capturing both local and global dependencies effectively. Unlike RNNs, self-attention can process all words in a sequence simultaneously, which makes training faster. Unlike CNNs, self-attention also doesn't have a fixed receptive field meaning it can look at more distant words.

## Positional Encoding

Unlike RNNs and CNNs, which would process sequences in order, the Transformer processes all words in parallel. This means that the model has no inherent way to understand word order, which is critical for tasks like translation and sentence structure comprehension. To combat this, Transformers use Positional Encoding-a mathematical function that encodes each word's position in the sequence into a numerical representation. Instead of using learned positional embeddings, the Transformer adds a position dependent vector to each word embedding before applying self-attention.

This is given by $PE_{(pos,2i) = sin(pos/10000^{2i/d_{model}}}$. 

where pos is the position of the word in the sequence, i is the dimension index, and $d_{model}$ is the total embedding size.

Sinusoidal and cosinusoidal functions are used to ensure a smooth wave pattern. These functions encode relative positions, meaning that the difference between any two positions is encoded consistently, making it easier for the model to generalize longer sequences. This doesn't requre training and generalizes to unseen sequence lengths since it is a mathematical function.

## Encoder-Decoder Architecture

The Transformer follows the encoder-decoder architecture, which is commonly used in machine translation, text summarization, and other NLP related tasks. 

The Transformer basically is made up of:

- Encoder: processes the input sequence and generates a contextual representation
- Decoder: takes the encoded representation and generates the output sequence

Unlike RNN-based encoder-decoder models, the Transformer will process the entire sequence.

### Encoder

The encoder is responsible for understanding the input sentences and making it into a meaningful representation. The encoder works by taking each input token (word) and converting into an embedding vector. Positional Encoding is added to retain word order information. The encoder is made up of N identical layers with each layer consisting of two main components:

- Multi-Head Self Attention: helps words attend to other words in the input sequence
- Feedforward Network: applies transformations to enhance feature extraction

The Transformer uses residual connections to help gradients flow through deep layers. Layer normalization stabilizes training by normalizing values at each layer. After the final encoder layer, we obtain a set of encoded representations for all input words.

### Decoder

The decoder is responsible for generating the output sequence step by step. It receives the encoder output and uses it as context for generating text. Each decoder laer has:

- Masked Multi-Head Self-Attention: prevents the decoder from "cheating" by looking at future tokens
- Multi-Head Attention Over Encoder Outputs: atends to encoded input representations
- Feedforward Network: similar to the encoder, it further processes the representations
- Residual Connections & Layer Normalization: improves gradient flow and stability

The auto-regressive generation is the decoder generating one word at a time. It feeds the previously generated words as input for predicting the next word. This ensures a natural sequence structure. The final output is the decoder outputting a probability distribution over the vocabulary and the model picks the most likely next word. This is fully parallelizable, better at long-range dependencies, and are scalable for modern NLP models like GPT and BERT.

