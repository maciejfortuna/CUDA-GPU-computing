#include "my_functions.h"
#include <cstring>
#include <fstream>
#include <random>
#include <iostream>
#include <iomanip>

void show_snake(sf::RenderWindow* wind, Snake snake)
{
	sf::RectangleShape rectangle;

	rectangle.setSize(sf::Vector2f(CELL_SIZE, CELL_SIZE));
	rectangle.setOutlineColor(sf::Color(0, 0, 0, 30));
	rectangle.setOutlineThickness(1);
	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{

			if (snake.map[i][j] == EMPTY)
			{
				rectangle.setFillColor(sf::Color(255, 255, 255));
			}
			if (snake.map[i][j] == SNAKE)
			{
				rectangle.setFillColor(sf::Color(0, 0, 0));
			}
			if (snake.map[i][j] == APPLE)
			{
				rectangle.setFillColor(sf::Color::Red);
			}
			if (snake.map[i][j] == -1)
			{
				rectangle.setFillColor(sf::Color::Blue);
			}
			rectangle.setPosition(j * CELL_SIZE, i * CELL_SIZE);
			wind->draw(rectangle);
		}
	}

	rectangle.setFillColor(sf::Color(224, 0, 183));
	rectangle.setPosition(snake.myPosesIDx[0] * CELL_SIZE, snake.myPosesIDy[0] * CELL_SIZE);
	wind->draw(rectangle);
}

void show_snake_gpu(sf::RenderWindow* wind, Snake_GPU snake)
{
	sf::RectangleShape rectangle;

	rectangle.setSize(sf::Vector2f(CELL_SIZE, CELL_SIZE));
	rectangle.setOutlineColor(sf::Color(0, 0, 0, 30));
	rectangle.setOutlineThickness(1);
	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{

			if (snake.map[i][j] == EMPTY)
			{
				rectangle.setFillColor(sf::Color(255, 255, 255));
			}
			if (snake.map[i][j] == SNAKE)
			{
				rectangle.setFillColor(sf::Color(0, 0, 0));
			}
			if (snake.map[i][j] == APPLE)
			{
				rectangle.setFillColor(sf::Color::Red);
			}
			if (snake.map[i][j] == -1)
			{
				rectangle.setFillColor(sf::Color::Blue);
			}
			rectangle.setPosition(j * CELL_SIZE, i * CELL_SIZE);
			wind->draw(rectangle);
		}
	}

	rectangle.setFillColor(sf::Color(224, 0, 183));
	rectangle.setPosition(snake.myPosesIDx[0] * CELL_SIZE, snake.myPosesIDy[0] * CELL_SIZE);
	wind->draw(rectangle);
}

void preview_snake(Snake* snk, sf::RenderWindow* wind)
{
	while (wind->isOpen() && snk->DEAD == 0)
	{

		snk->look_around();
		int out = snk->calculate_output();
		switch (out)
		{
			switch (out)
			{
			case 0:
				snk->dir -= 1;
				break;
			case 1:
				snk->dir += 0;
				break;
			case 2:
				snk->dir += 1;
				break;
			}
		}
		if (snk->dir < 0)
			snk->dir += 4;

		snk->dir = snk->dir % 4;
		snk->move(out);

		wind->clear();
		show_snake(wind, *snk);
		wind->display();
	}
}

Snake cross_over(Snake parent_0, Snake parent_1)
{

	Snake child;
	for (int i = 0; i < INPUT_WEIGHTS_SIZE; i++)
	{
		if (rand() % 2 == 0)
			child.input_weights[i] = parent_0.input_weights[i];
		else
			child.input_weights[i] = parent_1.input_weights[i];
	}
	for (int i = 0; i < HIDDEN_WEGHTS_SIZE; i++)
	{
		if (rand() % 2 == 0)
			child.hidden_weights[i] = parent_0.hidden_weights[i];
		else
			child.hidden_weights[i] = parent_1.hidden_weights[i];
	}
	
	for (int i = 0; i < OUTPUT_WEIGHTS_SIZE; i++)
	{
		if (rand() % 2 == 0)
			child.output_weights[i] = parent_0.output_weights[i];
		else
			child.output_weights[i] = parent_1.output_weights[i];
	}
	//--------------BIASES-------------------------
	for (int i = 0; i < HIDDEN_NEURONS_SIZE; i++)
	{
		if (rand() % 2 == 0)
			child.hidden_bias_0[i] = parent_0.hidden_bias_0[i];
		else
			child.hidden_bias_0[i] = parent_1.hidden_bias_0[i];
	}
	for (int i = 0; i < HIDDEN_NEURONS_1_SIZE; i++)
	{
		if (rand() % 2 == 0)
			child.hidden_bias_1[i] = parent_0.hidden_bias_1[i];
		else
			child.hidden_bias_1[i] = parent_1.hidden_bias_1[i];
	}
	
	for (int i = 0; i < OUTPUT_NEURONS_SIZE; i++)
	{
		if (rand() % 2 == 0)
			child.output_bias[i] = parent_0.output_bias[i];
		else
			child.output_bias[i] = parent_1.output_bias[i];
	}
	return child;
}

void mutate(Snake* snake)
{
	int r = 0;


	for (int i = 0; i < INPUT_WEIGHTS_SIZE; i++)
	{
		if (rand() % 100 <= MUTATION_RATE)
		{
			if (rand() % 2 == 0)
				r = 1;
			else
				r = -1;

			snake->input_weights[i] += MUTATION_SIZE * r;
		}
	}
	for (int i = 0; i < HIDDEN_WEGHTS_SIZE; i++)
	{
		if (rand() % 100 <= MUTATION_RATE)
		{
			if (rand() % 2 == 0)
				r = 1;
			else
				r = -1;
			snake->hidden_weights[i] += MUTATION_SIZE * r;
		}
	}
	
	for (int i = 0; i < OUTPUT_WEIGHTS_SIZE; i++)
	{
		if (rand() % 100 <= MUTATION_RATE)
		{
			if (rand() % 2 == 0)
				r = 1;
			else
				r = -1;
			snake->output_weights[i] += MUTATION_SIZE * r;
		}
	}
	//--------------BIASES-------------------------
	for (int i = 0; i < HIDDEN_NEURONS_SIZE; i++)
	{
		if (rand() % 100 <= MUTATION_RATE)
		{
			if (rand() % 2 == 0)
				r = 1;
			else
				r = -1;
			snake->hidden_bias_0[i] += MUTATION_SIZE * r;
		}
	}
	for (int i = 0; i < HIDDEN_NEURONS_1_SIZE; i++)
	{
		if (rand() % 100 <= MUTATION_RATE)
		{
			if (rand() % 2 == 0)
				r = 1;
			else
				r = -1;
			snake->hidden_bias_1[i] += MUTATION_SIZE * r;
		}
	}

	for (int i = 0; i < OUTPUT_NEURONS_SIZE; i++)
	{
		if (rand() % 100 <= MUTATION_RATE)
		{
			if (rand() % 2 == 0)
				r = 1;
			else
				r = -1;
			snake->output_bias[i] += MUTATION_SIZE * r;
		}
	}
}

Snake childess_parent(Snake parent_0)
{
	Snake child;
	//Cross Over Input Weights
	for (int i = 0; i < INPUT_WEIGHTS_SIZE; i++)
	{
		child.input_weights[i] = parent_0.input_weights[i];
	}
	for (int i = 0; i < HIDDEN_WEGHTS_SIZE; i++)
	{
		child.hidden_weights[i] = parent_0.hidden_weights[i];
	}
	for (int i = 0; i < OUTPUT_WEIGHTS_SIZE; i++)
	{
		child.output_weights[i] = parent_0.output_weights[i];
	}
	return child;
}

Snake load_snake(std::string path)
{
	int line = 0;
	int inp = 0;
	int hid = 0;

	int out = 0;

	int hid_0_bias = 0;
	int hid_1_bias = 0;
	int out_bias = 0;
	Snake player;

	std::ifstream file(path);
	std::string str;
	std::fstream myfile(path);
	float a;
	while (myfile >> a)
	{
		
		if (line < INPUT_WEIGHTS_SIZE)
		{
			player.input_weights[inp] = a;
			inp++;
		}
		if (line >= INPUT_WEIGHTS_SIZE && line < INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE)
		{
			player.hidden_weights[hid] = a;
			hid++;
		}

		if (line >= INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE && line < INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE)
		{
			player.output_weights[out] = a;
			out++;
		}

		if (line >= INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE && line < INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE + HIDDEN_NEURONS_SIZE)
		{
			player.hidden_bias_0[hid_0_bias] = a;
			hid_0_bias++;
		}
		if (line >= INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE + HIDDEN_NEURONS_SIZE && line < INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE + HIDDEN_NEURONS_SIZE + HIDDEN_NEURONS_1_SIZE)
		{

			player.hidden_bias_1[hid_1_bias] = a;
			hid_1_bias++;
		}

		if (line >= INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE + HIDDEN_NEURONS_SIZE + HIDDEN_NEURONS_1_SIZE && line < INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE + HIDDEN_NEURONS_SIZE + HIDDEN_NEURONS_1_SIZE + OUTPUT_NEURONS_SIZE)

		{

			player.output_bias[out_bias] = a;
			out_bias++;
			

		}

		line++;
	}



	myfile.close();
	file.close();
	return player;
}

Snake_GPU load_snake_gpu(std::string path)
{
	int line = 0;
	int inp = 0;
	int hid = 0;

	int out = 0;

	int hid_0_bias = 0;
	int hid_1_bias = 0;
	int out_bias = 0;
	Snake_GPU player;

	std::ifstream file(path);
	std::string str;
	std::fstream myfile(path);
	float a;
	while (myfile >> a)
	{

		if (line < INPUT_WEIGHTS_SIZE)
		{
			player.input_weights[inp] = a;
			inp++;
		}
		if (line >= INPUT_WEIGHTS_SIZE && line < INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE)
		{
			player.hidden_weights[hid] = a;
			hid++;
		}

		if (line >= INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE && line < INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE)
		{
			player.output_weights[out] = a;
			out++;
		}

		if (line >= INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE && line < INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE + HIDDEN_NEURONS_SIZE)
		{
			player.hidden_bias_0[hid_0_bias] = a;
			hid_0_bias++;
		}
		if (line >= INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE + HIDDEN_NEURONS_SIZE && line < INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE + HIDDEN_NEURONS_SIZE + HIDDEN_NEURONS_1_SIZE)
		{

			player.hidden_bias_1[hid_1_bias] = a;
			hid_1_bias++;
		}

		if (line >= INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE + HIDDEN_NEURONS_SIZE + HIDDEN_NEURONS_1_SIZE && line < INPUT_WEIGHTS_SIZE + HIDDEN_WEGHTS_SIZE + OUTPUT_WEIGHTS_SIZE + HIDDEN_NEURONS_SIZE + HIDDEN_NEURONS_1_SIZE + OUTPUT_NEURONS_SIZE)

		{

			player.output_bias[out_bias] = a;
			out_bias++;


		}

		line++;
	}



	myfile.close();
	file.close();
	return player;
}
void save_to_txt(Snake snk, std::string path, int gen_number)
{
	std::ofstream outfile(path + std::to_string(gen_number) + ".txt");

	for (int i = 0; i < INPUT_WEIGHTS_SIZE; i++)
	{
		outfile << std::setprecision(8);
		outfile << snk.input_weights[i] << std::endl;
	}
	for (int i = 0; i < HIDDEN_WEGHTS_SIZE; i++)
	{
		outfile << std::setprecision(8);

		outfile << snk.hidden_weights[i] << std::endl;
	}

	for (int i = 0; i < OUTPUT_WEIGHTS_SIZE; i++)
	{
		outfile << std::setprecision(8);

		outfile << snk.output_weights[i] << std::endl;
	}
	//------------------BIASES---------------------
	for (int i = 0; i < HIDDEN_NEURONS_SIZE; i++)
	{
		outfile << std::setprecision(8);

		outfile << snk.hidden_bias_0[i] << std::endl;
	}
	for (int i = 0; i < HIDDEN_NEURONS_1_SIZE; i++)
	{
		outfile << std::setprecision(8);

		outfile << snk.hidden_bias_1[i] << std::endl;
	}

	for (int i = 0; i < OUTPUT_NEURONS_SIZE; i++)
	{
		outfile << std::setprecision(8);

		outfile << snk.output_bias[i] << std::endl;
	}
	outfile.close();
}
void save_to_txt_gpu(Snake_GPU snk, std::string path, int gen_number)
{
	std::ofstream outfile(path + std::to_string(gen_number) + ".txt");

	for (int i = 0; i < INPUT_WEIGHTS_SIZE; i++)
	{
		outfile << std::setprecision(8);

		outfile << snk.input_weights[i] << std::endl;
	}
	for (int i = 0; i < HIDDEN_WEGHTS_SIZE; i++)
	{
		outfile << std::setprecision(8);

		outfile << snk.hidden_weights[i] << std::endl;
	}

	for (int i = 0; i < OUTPUT_WEIGHTS_SIZE; i++)
	{
		outfile << std::setprecision(8);

		outfile << snk.output_weights[i] << std::endl;
	}
	//------------------BIASES---------------------
	for (int i = 0; i < HIDDEN_NEURONS_SIZE; i++)
	{
		outfile << std::setprecision(8);

		outfile << snk.hidden_bias_0[i] << std::endl;
	}
	for (int i = 0; i < HIDDEN_NEURONS_1_SIZE; i++)
	{
		outfile << std::setprecision(8);

		outfile << snk.hidden_bias_1[i] << std::endl;
	}

	for (int i = 0; i < OUTPUT_NEURONS_SIZE; i++)
	{
		outfile << std::setprecision(8);

		outfile << snk.output_bias[i] << std::endl;
	}
	outfile.close();
}


double measure_and_show_time(clock_t clock_start, clock_t clock_end, std::string text)
{
	double czas = double(clock_end - clock_start) / double(CLOCKS_PER_SEC);
	return czas;
}