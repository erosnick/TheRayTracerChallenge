#pragma once

#include <cstdio>

inline size_t getFileLength(FILE* file)
{
	size_t fileLength = 0;

	if (file != nullptr)
	{
		fseek(file, 0, SEEK_END);

		fileLength = ftell(file);

		fseek(file, 0, SEEK_CUR);
	}

	return fileLength;
}