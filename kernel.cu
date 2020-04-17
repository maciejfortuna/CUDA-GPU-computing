
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/device_ptr.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <vector>
#include <stdlib.h>
#include <fstream>
#include <string>
#include "my_functions.h"


__global__ void calc_out_kernel(Snake_GPU* player)
{

	curandState state;
	curand_init((unsigned long long)clock(), 0, 0, &state);
	player->local_state = state;
	player->look_around();
	int out = player->calculate_output();
	player->move(out);
}

#if EVOLVE_ON_GPU

__global__ void setup_cuRand(curandState* state)
{
	int id = threadIdx.x + blockDim.x + blockIdx.x;
	if (id < POP_SIZE)
		curand_init(1234, id, 0, &state[id]);
}

__global__ void evaluate_fintess_kernell(Snake_GPU* dev_snakes, float* dev_fit, curandState* st, int init_cur)
{
	int threadID = threadIdx.x + blockDim.x * blockIdx.x;
	int krok = blockDim.x * gridDim.x;
	if (threadID < POP_SIZE && init_cur == 0)
	{
		dev_snakes[threadID].local_state = st[threadID];
	}
	for (int i = threadID; i < POP_SIZE; i += krok)
	{
		dev_snakes[i].play_game();
		dev_fit[i] = dev_snakes[i].fitness;
	}
}

__global__ void crossover_kernel(Snake_GPU* dev_snakes, Snake_GPU* best_snakes, float* dev_fit, curandState* st)
{
	int threadID_x = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (threadID_x < POP_SIZE)
	{
		int krok_x = blockDim.x * gridDim.x;

		if (threadID_x == 0)
		{
			printf("%f;%d;", dev_fit[0], best_snakes[0].score);
		}

		int j = threadID_x;
		curandState state;
		curand_init(1234, threadID_x, 0, &state);

			dev_snakes[j].reset();
			int first = curand(&state) % BEST_SNAKES_SIZE;
			int second = curand(&state) % (BEST_SNAKES_SIZE - 1);

			//printf("which: %d || FIRST: %d || SECOND: %d \n",j, first, second);

			//WAGI
			for (int i = 0; i < INPUT_WEIGHTS_SIZE; i += 1)
			{
				if (curand(&state) % 2 == 0)
					dev_snakes[j].input_weights[i] = best_snakes[first].input_weights[i];
				else
					dev_snakes[j].input_weights[i] = best_snakes[second].input_weights[i];
			}
			for (int i = 0; i < HIDDEN_WEGHTS_SIZE; i += 1)
			{
				if (curand(&state) % 2 == 0)
					dev_snakes[j].hidden_weights[i] = best_snakes[first].hidden_weights[i];
				else
					dev_snakes[j].hidden_weights[i] = best_snakes[second].hidden_weights[i];
			}

			for (int i = 0; i < OUTPUT_WEIGHTS_SIZE; i += 1)
			{
				if (curand(&state) % 2 == 0)
					dev_snakes[j].output_weights[i] = best_snakes[first].output_weights[i];
				else
					dev_snakes[j].output_weights[i] = best_snakes[second].output_weights[i];
			}

		
			for (int i = 0; i < HIDDEN_NEURONS_SIZE; i += 1)
			{
				if (curand(&state) % 2 == 0)
					dev_snakes[j].hidden_bias_0[i] = best_snakes[first].hidden_bias_0[i];
				else
					dev_snakes[j].hidden_bias_0[i] = best_snakes[second].hidden_bias_0[i];
			}
			for (int i = 0; i < HIDDEN_NEURONS_1_SIZE; i += 1)
			{
				if (curand(&state) % 2 == 0)
					dev_snakes[j].hidden_bias_1[i] = best_snakes[first].hidden_bias_1[i];
				else
					dev_snakes[j].hidden_bias_1[i] = best_snakes[second].hidden_bias_1[i];
			}

			for (int i = 0; i < OUTPUT_NEURONS_SIZE; i += 1)
			{
				if (curand(&state) % 2 == 0)
					dev_snakes[j].output_bias[i] = best_snakes[first].output_bias[i];
				else
					dev_snakes[j].output_bias[i] = best_snakes[second].output_bias[i];
			}

			//MUTACJA
			int r = 0;
			//WAGI
			for (int i = 0; i < INPUT_WEIGHTS_SIZE; i += 1)
			{
				if (curand(&state) % 100 <= MUTATION_RATE)
				{
					if (curand(&state) % 2 == 0)
						r = 1;
					else
						r = -1;

					dev_snakes[j].input_weights[i] += MUTATION_SIZE * r;
				}
			}
			for (int i = 0; i < HIDDEN_WEGHTS_SIZE; i += 1)
			{
				if (curand(&state) % 100 <= MUTATION_RATE)
				{
					if (curand(&state) % 2 == 0)
						r = 1;
					else
						r = -1;
					dev_snakes[j].hidden_weights[i] += MUTATION_SIZE * r;
				}
			}

			for (int i = 0; i < OUTPUT_WEIGHTS_SIZE; i += 1)
			{
				if (curand(&state) % 100 <= MUTATION_RATE)
				{
					if (curand(&state) % 2 == 0)
						r = 1;
					else
						r = -1;
					dev_snakes[j].output_weights[i] += MUTATION_SIZE * r;
				}
			}

			//BIASY
			for (int i = 0; i < HIDDEN_NEURONS_SIZE; i += 1)
			{
				if (curand(&state) % 100 <= MUTATION_RATE)
				{
					if (curand(&state) % 2 == 0)
						r = 1;
					else
						r = -1;
					dev_snakes[j].hidden_bias_0[i] += MUTATION_SIZE * r;
				}
			}
			for (int i = 0; i < HIDDEN_NEURONS_1_SIZE; i += 1)
			{
				if (curand(&state) % 100 <= MUTATION_RATE)
				{
					if (curand(&state) % 2 == 0)
						r = 1;
					else
						r = -1;
					dev_snakes[j].hidden_bias_1[i] += MUTATION_SIZE * r;
				}

			}

			for (int i = 0; i < OUTPUT_NEURONS_SIZE; i += 1)
			{
				if (curand(&state) % 100 <= MUTATION_RATE)
				{
					if (curand(&state) % 2 == 0)
						r = 1;
					else
						r = -1;
					dev_snakes[j].output_bias[i] += MUTATION_SIZE * r;
				}
			}

		}
	
}

#endif

int main()
{
	bool load_from_file = true;
	std::string path_load = "SAVED/GPU_26_01_2020_12x12/GENERATION_1767.txt";

	bool use_keyboard = false;
	bool train_on_cpu = false;

	clock_t start_all_generations, end_all_generations;
	clock_t start_single_gen, end_single_gen;
	clock_t start_mut_and_cross, end_mut_and_cross;

	srand(time(NULL));

#if USE_CPU
	Snake player;

	//GRAFIKA
	sf::RenderWindow window(sf::VideoMode(WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE), "SFML works!");
	sf::RenderWindow side_window(sf::VideoMode(200, 400), "SFML works!");
	side_window.setPosition(window.getPosition() + sf::Vector2i(600, 0));
	sf::Font font;
	font.loadFromFile("fonts/arial.ttf");
	sf::Text scoreText;
	scoreText.setFont(font);
	scoreText.setCharacterSize(12);
	scoreText.setFillColor(sf::Color::White);
	scoreText.setPosition(0, 0);

	if (load_from_file)
	{

		Snake_GPU host_snake = load_snake_gpu(path_load);
		Snake_GPU* dev_snake = 0;
		Snake_GPU player_gpu;
		cudaMalloc(&dev_snake, sizeof(Snake_GPU));
		cudaMemcpy(dev_snake, &host_snake, sizeof(Snake_GPU), cudaMemcpyHostToDevice);

		while (player_gpu.DEAD == 0)
		{
			calc_out_kernel << <1, 1 >> > (dev_snake);
			cudaDeviceSynchronize();

			cudaMemcpy(&player_gpu, dev_snake, sizeof(Snake_GPU), cudaMemcpyDeviceToHost);

			scoreText.setString("SCORE: " + std::to_string(player_gpu.score) + "\nSTEPS_TAKEN: " + std::to_string(player_gpu.steps_taken)
				+ "\nSTEPS_LEFT: " + std::to_string(player_gpu.steps_left));

			window.clear();
			side_window.clear();
			show_snake_gpu(&window, player_gpu);
			side_window.draw(scoreText);
			window.display();
			side_window.display();

		}

		/*player = load_snake(path_load);
		while (player.dead == 0)
		{
			
			player.look_around();
			int out = player.calculate_output();
			player.move(out);

			scoretext.setstring("score: " + std::to_string(player.score) + "\nsteps_taken: " + std::to_string(player.steps_taken)
				+ "\nsteps_left: " + std::to_string(player.steps_left));

			window.clear();
			side_window.clear();
			show_snake(&window, player);
			side_window.draw(scoretext);
			window.display();
			side_window.display();


		}*/
	}

	if (use_keyboard)
	{
		while (window.isOpen() && player.DEAD == 0)
		{
			sf::Event event;
			while (window.pollEvent(event))
			{
				switch (event.type)
					{
						// window closed
					case sf::Event::Closed:
						window.close();
						break;

						// key pressed
					case sf::Event::KeyPressed:
						if (event.key.code == sf::Keyboard::W)
						{
							player.dir = UP;
						}
						if (event.key.code == sf::Keyboard::D)
						{
							player.dir = RIGHT;
						}
						if (event.key.code == sf::Keyboard::S)
						{
							player.dir = DOWN;
						}
						if (event.key.code == sf::Keyboard::A)
						{
							player.dir = LEFT;
						}


						player.move(-10);

						player.look_around();
					}
				}

				window.clear();
				show_snake(&window, player);
				window.display();
			}
		}

	if (train_on_cpu)
	{
		start_all_generations = clock();
		Snake temp;
		std::vector<Snake> snakes(POP_SIZE);
		std::vector<Snake> best_snakes;

		int out = 0;
		int first = 0;
		int second = 0;

		std::ofstream logs("SAVED/CPU/LOGS.txt");
		logs << "GENERATION;BEST_FITNESS;BEST_SCORE;GENERATION_TIME" << std::endl;
		printf("GENERATION;BEST_FITNESS;BEST_SCORE;GENERATION_TIME \n");

		for (int i = 0; i < GENERATION; i++)
		{
			start_single_gen = clock();

			

			for (int i = 0; i < POP_SIZE; i++)
			{
				snakes.at(i).id = i;
				snakes.at(i).play_game();
			}

			std::sort(snakes.begin(), snakes.end(), [](const Snake& lhs, const Snake& rhs)
			{
					return lhs.fitness > rhs.fitness;
			});

			for (int i = 0; i < BEST_SNAKES_SIZE; i++)
			{
				best_snakes.push_back(snakes.at(i));
			}
			save_to_txt(best_snakes.at(0), "SAVED/CPU/GENERATION_", i);

			snakes.clear();

			start_mut_and_cross = clock();
			for (int i = 0; i < POP_SIZE; i++)
			{
				
				first = rand() % BEST_SNAKES_SIZE;
				second = rand() % (BEST_SNAKES_SIZE - 1);
				if (second >= first)
				{
					++second;
				}
				if (rand() % 100 <= CHILDESS_PARENT_PROB)
				{
					temp = childess_parent(best_snakes.at(0));
				}
				else
				{
					temp = cross_over(best_snakes.at(first), best_snakes.at(second));
					mutate(&temp);
				}
				snakes.push_back(temp);
			}
			end_mut_and_cross = clock();
			end_single_gen = clock();
			double czas = measure_and_show_time(start_single_gen, end_single_gen, "this generation time: ");
			logs << i <<";" << best_snakes.at(0).fitness << ";"<<best_snakes.at(0).score << ";" << czas<< std::endl;
			printf("%d;%f;%d;%f \n",i, best_snakes.at(0).fitness, best_snakes.at(0).score, czas);

			best_snakes.clear();

			
		}
		logs.close();

		end_all_generations = clock();
		measure_and_show_time(start_all_generations, end_all_generations, "all simulation time: ");
	}

#endif
		
#if EVOLVE_ON_GPU
			start_all_generations = clock();

			std::ofstream logs("SAVED/GPU/LOGS.txt");
			thrust::host_vector<Snake_GPU> h_snakes(POP_SIZE);
			thrust::host_vector<Snake_GPU> h_best_snakes(BEST_SNAKES_SIZE);

			thrust::device_vector<int> ids(POP_SIZE);
			thrust::device_vector<float> fitnesses(POP_SIZE);


			int block_size = 512;
			int num_blocks = (POP_SIZE + block_size - 1) / block_size;
			curandState* randState;
			int myCurandSize = POP_SIZE;
			cudaMalloc((void**)&randState, myCurandSize * sizeof(curandState));
			setup_cuRand << <num_blocks, block_size >> > (randState);
			cudaDeviceSynchronize();

			for (int i = 0; i < POP_SIZE; i++)
			{
				h_snakes[i].id = i;

			}

			thrust::device_vector<Snake_GPU> best_snakes_dev = h_best_snakes;
			thrust::device_vector<Snake_GPU> d_snakes = h_snakes;

			Snake_GPU* dev_ptr = thrust::raw_pointer_cast(&d_snakes[0]);
			float* dev_ptr_fit = thrust::raw_pointer_cast(&fitnesses[0]);
			Snake_GPU* dev_ptr_best_snakes = thrust::raw_pointer_cast(&best_snakes_dev[0]);

		
			dim3 grid(1000, 1, 1);
			dim3 block(7, 120, 1);

			logs << "GENERATION;BEST_FITNESS;BEST_SCORE;GENERATION_TIME" << std::endl;
			printf("GENERATION;BEST_FITNESS;BEST_SCORE;GENERATION_TIME \n");


			int init_cur = 0;
			for (int i = 0; i < GENERATION; i++)
			{
				start_single_gen = clock();
				printf("%d;", i);

				evaluate_fintess_kernell << < 10000, 512 >> > (dev_ptr, dev_ptr_fit, randState,init_cur);
				cudaDeviceSynchronize();
				init_cur = 1;


				thrust::sequence(thrust::device, ids.begin(), ids.end(), 0);
				thrust::sort_by_key(thrust::device, fitnesses.begin(), fitnesses.end(), ids.begin(), thrust::greater<float>());
				cudaDeviceSynchronize();

				for (int i = 0; i < BEST_SNAKES_SIZE; i++)
				{
					best_snakes_dev[i] = d_snakes[ids[i]];
				}


				save_to_txt_gpu(best_snakes_dev[0], "SAVED/GPU_26_01_2020_12x12/GENERATION_", i);

				start_mut_and_cross = clock();

				crossover_kernel << < 10000, 512 >> > (dev_ptr, dev_ptr_best_snakes, dev_ptr_fit,randState);
				cudaDeviceSynchronize();

				end_mut_and_cross = clock();
				end_single_gen = clock();

				//measure_and_show_time(start_mut_and_cross, end_mut_and_cross, "mutation and cross time: ");
				double czas = measure_and_show_time(start_single_gen, end_single_gen, "this generation time: ");
				printf("%f \n", czas);
				//logs << i << ";" << best_snakes.at(0).fitness << ";" << best_snakes.at(0).score << ";" << czas << std::endl;
	
			}
			logs.close();
			end_all_generations = clock();
			measure_and_show_time(start_all_generations, end_all_generations, "all simulation time: ");
#endif




    return 0;
}