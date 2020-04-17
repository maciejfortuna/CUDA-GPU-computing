#pragma once
#include "snake_class.h"
#include "snake_class_gpu.h"
void show_snake(sf::RenderWindow* wind, Snake snake);
void show_snake_gpu(sf::RenderWindow* wind, Snake_GPU snake);
void preview_snake(Snake* snk, sf::RenderWindow* wind);
Snake cross_over(Snake parent_0, Snake parent_1);
void mutate(Snake* snake);
Snake load_snake(std::string path);
Snake_GPU load_snake_gpu(std::string path);
Snake childess_parent(Snake parent_0);
void save_to_txt(Snake snk, std::string path, int gen_number);
void save_to_txt_gpu(Snake_GPU snk, std::string path, int gen_number);
double measure_and_show_time(clock_t clock_start, clock_t clock_end, std::string text);