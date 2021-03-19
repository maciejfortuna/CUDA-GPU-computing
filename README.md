# CUDA/GPU parallel computing
Implementation of genetic algorithm for parallel computing using CUDA/C++ that finds best weights and biases of artificial neural network that plays snake game.

Neural network is programmed from scratch (multiplication and summation of neurons, input, output etc.)

Evaluation, crossover and mutation are done using parallel computing on graphic card (GPU) to increase computation time.

I could not use structures like queue, vectors etc. because CUDA is not compatible with it, so I had to operate on basic arrays and implement many functions myself.

## RESULTS
![gif](https://github.com/maciejfortuna/CUDA-GPU-computing/blob/master/gif_result.gif)
## GPU
![GPU](https://github.com/maciejfortuna/CUDA-GPU-computing/blob/master/GPU_Graph.png)
## GPU vs CPU
![GPU vs CPU](https://github.com/maciejfortuna/CUDA-GPU-computing/blob/master/GPU%20vs%20CPU.png)
