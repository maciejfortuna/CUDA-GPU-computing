#pragma once
#include "constants.h"
#include <random>
#include "SFML/Graphics.hpp"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "math.h"
#include "device_launch_parameters.h"

class Snake_GPU {
public:
	int myPosesIDx[WIDTH * HEIGHT] = { -1 };
	int myPosesIDy[WIDTH * HEIGHT] = { -1 };
	int id;
	int map[HEIGHT][WIDTH];
	int score;
	int dir = UP;
	int DEAD = 0;
	int steps_left = 100;
	int steps_taken = 0;
	float fitness = 0;
	int avl_indexes_x[WIDTH * HEIGHT] = { -1 };
	int avl_indexes_y[HEIGHT * HEIGHT] = { -1 };

	//--------------------NEURAL NETWORK LOOK DIRECTIONS-----------------------
	float food_dist[LOOK_DIRECTIONS] = { -1.0 };
	float body_dist[LOOK_DIRECTIONS] = { -1.0 };
	float wall_dist[LOOK_DIRECTIONS] = { -1.0 };
	//--------------------NEURAL NETWORK NEURONS-----------------------
	float input[INPUT_SIZE];
	float hidden_neurons_0[HIDDEN_NEURONS_SIZE] = { 0.0 };
	float hidden_neurons_1[HIDDEN_NEURONS_1_SIZE] = { 0.0 };
	
	float output[OUTPUT_NEURONS_SIZE] = { 0.0 };
	//--------------------NEURAL NETWORK WEIGHTS-----------------------
	float input_weights[INPUT_WEIGHTS_SIZE] = { 1.0 };
	float hidden_weights[HIDDEN_WEGHTS_SIZE] = { 1.0 };
	
	float output_weights[OUTPUT_WEIGHTS_SIZE] = { 1.0 };
	//--------------------NEURAL NETWORK BIASES-----------------------
	float hidden_bias_0[HIDDEN_NEURONS_SIZE] = { 0.0 };
	float hidden_bias_1[HIDDEN_NEURONS_1_SIZE] = { 0.0 };

	float output_bias[OUTPUT_NEURONS_SIZE] = { 0.0 };

	curandState local_state;
	__host__
	Snake_GPU();
	__device__
	void move(int out);
	__device__
	void look_around();
	__device__
	float fast_sigmoid(float x);
	__device__
	float relu_activation(float x);
	
	__device__
	int calculate_output();
	__device__
	void generate_apple();
	__host__
	void generate_apple_host();
	__device__
	void end_game();
	
	__device__
	void play_game();

	__device__
	void reset();
	__device__ void draw_map();
};
