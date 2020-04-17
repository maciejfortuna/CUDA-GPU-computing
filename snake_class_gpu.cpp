#pragma once
#include "snake_class_gpu.h"

	__host__
	Snake_GPU::Snake_GPU()
	{
		score = 4;
		for (int i = 0; i < HEIGHT * WIDTH; i++)
		{
			myPosesIDx[i] = -1;
			myPosesIDy[i] = -1;
		}
		for (int i = 0; i < 4; i++)
		{
			myPosesIDx[i] = WIDTH / 2;
			myPosesIDy[i] = HEIGHT / 2 + i;
		}
		//Inicjalizacja MAPY
		for (int i = 0; i < HEIGHT; i++)
		{
			for (int j = 0; j < WIDTH; j++)
			{
				map[i][j] = EMPTY;
			}
		}
		for (int i = 0; i < WIDTH * HEIGHT; i++)
		{
			int x = myPosesIDx[i];
			int y = myPosesIDy[i];
			if (x != -1 && y != -1)
				map[y][x] = SNAKE;
		}
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> distribution(-5.0, 5.0);
		for (int i = 0; i < INPUT_WEIGHTS_SIZE; i++)
		{
			input_weights[i] = distribution(gen);
		}
		for (int i = 0; i < HIDDEN_WEGHTS_SIZE; i++)
		{
			hidden_weights[i] = distribution(gen);
		}
		
		for (int i = 0; i < OUTPUT_WEIGHTS_SIZE; i++)
		{
			output_weights[i] = distribution(gen);
		}
		//----------------BIASES
		for (int i = 0; i < HIDDEN_NEURONS_SIZE; i++)
		{
			hidden_bias_0[i] = distribution(gen);
		}
		for (int i = 0; i < HIDDEN_NEURONS_1_SIZE; i++)
		{
			hidden_bias_1[i] = distribution(gen);
		}
		
		for (int i = 0; i < OUTPUT_NEURONS_SIZE; i++)
		{
			output_bias[i] = distribution(gen);
		}

		generate_apple_host();

	}

	__device__
	Snake_GPU::move(int out)
	{
	
		steps_left--;
		if (steps_left < 0)
		{
			end_game();
			return;
		}
		steps_taken++;
		int firstX = myPosesIDx[0];
		int firstY = myPosesIDy[0];

		int nextPosX = firstX;
		int nextPosY = firstY;
		if (out != -10)
		{
			switch (out)
			{

			case 0:
				dir -= 1;
				break;
			case 1:
				dir += 0;
				break;
			case 2:
				dir += 1;
				break;

			}

			if (dir < 0)
				dir = 4 + dir;

			dir = dir % 4;
		}

		if (dir == UP)
		{
			nextPosY -= 1;
		}
		if (dir == RIGHT)
		{
			nextPosX += 1;
		}
		if (dir == DOWN)
		{
			nextPosY += 1;
		}
		if (dir == LEFT)
		{
			nextPosX -= 1;
		}
		if (nextPosX == WIDTH || nextPosX == -1 || nextPosY == HEIGHT || nextPosY == -1)
		{
			end_game();
			return;
		}
		else if (map[nextPosY][nextPosX] == SNAKE)
		{
			end_game();
			return;
		}

		if (map[nextPosY][nextPosX] != APPLE)
		{
			map[myPosesIDy[score - 1]][myPosesIDx[score - 1]] = EMPTY;
			myPosesIDx[score - 1] = -1;
			myPosesIDy[score - 1] = -1;
			for (int i = HEIGHT * WIDTH - 1; i > 0; i--)
			{
				myPosesIDx[i] = myPosesIDx[i - 1];
				myPosesIDy[i] = myPosesIDy[i - 1];
			}

			myPosesIDx[0] = nextPosX;
			myPosesIDy[0] = nextPosY;
			map[nextPosY][nextPosX] = SNAKE;
		}
		else
		{
			score++;

			if (score == WIDTH * HEIGHT)
			{
				end_game();
				return;
			}
			steps_left += 100;
			for (int i = HEIGHT * WIDTH - 1; i > 0; i--)
			{
				myPosesIDx[i] = myPosesIDx[i - 1];
				myPosesIDy[i] = myPosesIDy[i - 1];
			}

			myPosesIDx[0] = nextPosX;
			myPosesIDy[0] = nextPosY;
			map[nextPosY][nextPosX] = SNAKE;

			generate_apple();

		}
	}
	
	__device__
	Snake_GPU::look_around()
	{
#if VISION_4

		int h_x = myPosesIDx[0];
		int h_y = myPosesIDy[0];
		float food_y = -1;
		float food_minus_y = -1;
		float food_x = -1;
		float food_minus_x = -1;
		float body_y = -1;
		float body_minus_y = -1;
		float body_x = -1;
		float body_minus_x = -1;

		
		for (int i = h_y + 1; i < HEIGHT; i++)
		{
			if (map[i][h_x] == APPLE && body_minus_y == -1)
			{
				food_minus_y = i - h_y;
			}
			if (map[i][h_x] == SNAKE && body_minus_y == -1 && food_minus_y == -1)
			{
				body_minus_y = i - h_y;
			}
		}
		
		for (int i = h_y - 1; i >= 0; i--)
		{
			if (map[i][h_x] == APPLE && body_y == -1)
			{
				food_y = h_y - i;
			}
			if (map[i][h_x] == SNAKE && body_y == -1 && food_y == -1)
			{
				body_y = h_y - i;
			}
		}
		
		for (int i = h_x + 1; i < WIDTH; i++)
		{
			if (map[h_y][i] == APPLE && body_x == -1)
			{
				food_x = i - h_x;
			}
			if (map[h_y][i] == SNAKE && body_x == -1 && food_x == -1)
			{
				body_x = i - h_x;

			}
		}
		
		for (int i = h_x - 1; i >= 0; i--)
		{
			if (map[h_y][i] == APPLE && body_minus_x == -1)
			{
				food_minus_x = h_x - i;

			}
			if (map[h_y][i] == SNAKE && body_minus_x == -1 && food_minus_x == -1)
			{
				body_minus_x = h_x - i;
			}
		}
		int index = 0;

		switch (dir)
		{
		case UP:

			break;
		case RIGHT:
			index = 3;
			break;
		case DOWN:
			index = 2;
			break;
		case LEFT:
			index = 1;
			break;
		}

		int y = index;
		int x = (index + 1) % 4;
		int min_y = (x + 1) % 4;
		int min_x = (min_y + 1) % 4;

		food_dist[y] = food_y;
		body_dist[y] = body_y;
		wall_dist[y] = h_y + 1;

		food_dist[x] = food_x;
		body_dist[x] = body_x;
		wall_dist[x] = WIDTH - h_x;

		food_dist[min_y] = food_minus_y;
		body_dist[min_y] = body_minus_y;
		wall_dist[min_y] = HEIGHT - h_y;

		food_dist[min_x] = food_minus_x;
		body_dist[min_x] = body_minus_x;
		wall_dist[min_x] = h_x + 1;

		for (int i = 0; i < INPUT_SIZE; i++)
		{
			if (i < 4)
				input[i] = food_dist[i];
			if (i >= 4 && i < 8)
				input[i] = body_dist[i - 4];
			if (i >= 8 && i < 12)
				input[i] = wall_dist[i - 8];
		}
#endif


	}
	
	__device__
	Snake_GPU::fast_sigmoid(float x)
	{
		return x / (1 + abs(x));
	}
	__device__
	Snake_GPU::relu_activation(float x)
	{
		float y;
		if (x < 0)
			y = 0;
		else
			y = x;
		return y;
	}
	__device__
	Snake_GPU::calculate_output()
	{
		for (int i = 0; i < HIDDEN_NEURONS_SIZE; i++)
		{
			for (int j = 0; j < INPUT_SIZE; j++)
			{
				hidden_neurons_0[i] += input[j] * input_weights[i * HIDDEN_NEURONS_SIZE + j];
			}
			hidden_neurons_0[i] = fast_sigmoid(hidden_neurons_0[i] + hidden_bias_0[i]);
		}
		for (int i = 0; i < HIDDEN_NEURONS_1_SIZE; i++)
		{
			for (int j = 0; j < HIDDEN_NEURONS_SIZE; j++)
			{
				hidden_neurons_1[i] += hidden_neurons_0[j] * hidden_weights[i * HIDDEN_NEURONS_1_SIZE + j];
			}
			hidden_neurons_1[i] = fast_sigmoid(hidden_neurons_1[i] + hidden_bias_1[i]);
		}

		for (int i = 0; i < OUTPUT_NEURONS_SIZE; i++)
		{
			for (int j = 0; j < HIDDEN_NEURONS_1_SIZE; j++)
			{
				output[i] += hidden_neurons_1[j] * output_weights[i * OUTPUT_NEURONS_SIZE + j];
			}
			output[i] = fast_sigmoid(output[i] + output_bias[i]);
		}

		float max = output[0];
		int max_index = 0;
		for (int i = 1; i < OUTPUT_NEURONS_SIZE; i++)
		{
			if (output[i] > max)
			{
				max = output[i];
				max_index = i;
			}
		}
		return max_index;
	}

	
	__device__
	Snake_GPU::generate_apple()
	{
		int avl_size = WIDTH * HEIGHT - score;
		int k = -1;
		for (int i = 0; i < HEIGHT; i++)
		{
			for (int j = 0; j < WIDTH; j++)
			{
				if (map[i][j] == EMPTY)
				{
					k++;
					avl_indexes_x[k] = j;
					avl_indexes_y[k] = i;
				}
			}
		}
		
		int rand_index = curand(&local_state) % avl_size;
		
		map[avl_indexes_y[rand_index]][avl_indexes_x[rand_index]] = APPLE;
	}
	__host__
		Snake_GPU::generate_apple_host()
	{
		int avl_size = WIDTH * HEIGHT - score;
		int k = -1;
		for (int i = 0; i < HEIGHT; i++)
		{
			for (int j = 0; j < WIDTH; j++)
			{
				if (map[i][j] == EMPTY)
				{
					k++;
					avl_indexes_x[k] = j;
					avl_indexes_y[k] = i;
				}
			}
		}
		int rand_index = rand() % avl_size;
		map[avl_indexes_y[rand_index]][avl_indexes_x[rand_index]] = APPLE;
	}

	//KONCZY GRE,EWALUACJA FTINESS WEZA 
	__device__
		Snake_GPU::end_game()
	{
		fitness = steps_taken + (powf(2, score) + powf(score, 2.1) * 500) - (powf(score, 1.2) * powf(0.25 * steps_taken, 1.3));
		//fitness = score;
		DEAD = 1;
		return;
	}
	__device__
		Snake_GPU::play_game()
	{

		while (DEAD == 0)
		{
			look_around();
			int out = calculate_output();

			move(out);
		}
		return;
	}
	__device__
	Snake_GPU::reset()
	{
		DEAD = 0;
		score = 4;
		dir = UP;
		steps_left = 100;
		steps_taken = 0;
		fitness = 0;
		for (int i = 0; i < HEIGHT * WIDTH; i++)
		{
			myPosesIDx[i] = -1;
			myPosesIDy[i] = -1;
		}
		for (int i = 0; i < 4; i++)
		{
			myPosesIDx[i] = WIDTH / 2;
			myPosesIDy[i] = HEIGHT / 2 + i;
		}
		//Inicjalizacja MAPY
		for (int i = 0; i < HEIGHT; i++)
		{
			for (int j = 0; j < WIDTH; j++)
			{
				map[i][j] = EMPTY;
			}
		}
		for (int i = 0; i < WIDTH * HEIGHT; i++)
		{
			int x = myPosesIDx[i];
			int y = myPosesIDy[i];
			if (x != -1 && y != -1)
				map[y][x] = SNAKE;
		}

		generate_apple();
	}

	__device__
    Snake_GPU::draw_map()
	{
		for (int i = 0; i < WIDTH; i++)
		{
			printf("\n");
			for (int j = 0; j < HEIGHT; j++)
			{
				printf(" ");
				printf("%d", map[i][j]);
			}
		}
	}
};
