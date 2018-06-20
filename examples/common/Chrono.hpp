#pragma once

#include <chrono>
#include <string>

struct Chrono
{
	Chrono() :
		last( std::chrono::system_clock::now() )
	{ }

	void printAndReset( std::string eventName)
	{
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-last;
		std::cout << eventName << ":\t" << elapsed_seconds.count() << " s\n";
		last = end;
	}

	std::chrono::time_point< std::chrono::system_clock > last;
};
