#pragma once

#include <chrono>

typedef std::chrono::high_resolution_clock timer_clock;
typedef std::chrono::high_resolution_clock::time_point timer_tp;

timer_tp timer_now()
{
	return timer_clock::now();
}

float timer_elapsed(const timer_tp &start, const timer_tp &end)
{
	return static_cast<float>(std::chrono::duration<double, std::milli>(end - start).count());
}