#pragma once

#include "CUDA.h"
#include "FileUtils.h"
#include "Memory.h"
#include "Constants.h"

#include <iostream>
#include <stdint.h>
#include <cassert>

//template<typename T>
//class Array
//{
//public:
//	CUDA_HOST_DEVICE Array() {
//		internalCapacity = capacityIncrement;
//		internalArray = new T[internalCapacity];
//	}
//
//	CUDA_HOST_DEVICE Array(int32_t inSize) {
//		internalSize = inSize;
//
//		resize(internalSize);
//	}
//
//	CUDA_HOST_DEVICE ~Array() {
//		delete[] internalArray;
//	}
//
//	CUDA_HOST_DEVICE void add(T value) {
//        internalArray[internalSize] = value;
//        internalSize++;
//
//        if (internalSize == internalCapacity) {
//            resize(internalCapacity + capacityIncrement);
//        }
//	}
//
//	CUDA_HOST_DEVICE void remove(T value) {
//		int32_t foundIndex = -1;
//
//		for (auto i = 0; i < internalSize; i++) {
//			if (internalArray[i] == value) {
//				foundIndex = i;
//				break;
//			}
//		}
//
//        if (foundIndex >= 0) {
//            for (auto i = foundIndex; i < internalSize - 1; i++) {
//                internalArray[i] = internalArray[i + 1];
//            }
//
//            internalSize--;
//        }
//	}
//
//	CUDA_HOST_DEVICE void resize(int32_t newSize) {
//		if (internalCapacity == 0) {
//			internalCapacity = newSize + capacityIncrement;
//			internalArray = new T[internalCapacity]();
//			internalSize = newSize;
//		}
//		else if (newSize > internalCapacity) {
//			T* swapBuffer = new T[internalCapacity];
//
//			//memcpy_s(swapBuffer, sizeof(T) * internalCapacity, internalArray, sizeof(T) * internalSize);
//			Memory::memcpy(swapBuffer, internalArray, sizeof(T) * internalSize);
//
//			delete[] internalArray;
//
//			internalArray = swapBuffer;
//
//			swapBuffer = nullptr;
//
//			internalCapacity = newSize;
//		}
//	}
//
//	CUDA_HOST_DEVICE bool writeFile(const char* fileName) {
//		FILE* outFile = nullptr;
//
//		fopen_s(&outFile, fileName, "wb");
//
//		if (outFile == 0) {
//			printf("Open file %s failed.\n"), fileName;
//			return false;
//		}
//
//		size_t written = fwrite(internalArray, sizeof(T), internalSize, outFile);
//
//		if (written != internalSize) {
//			printf("Write file failed.\n");
//			return false;
//		}
//
//		fclose(outFile);
//
//		return true;
//	}
//
//	CUDA_HOST_DEVICE bool readFile(const char* fileName) {
//		FILE* inFile = nullptr;
//
//		fopen_s(&inFile, fileName, "rb");
//
//		if (inFile == nullptr) {
//			printf("Open file failed.\n");
//			return false;
//		}
//
//		fseek(inFile, 0, SEEK_END);
//
//		size_t fileLength = ftell(inFile);
//
//		fseek(inFile, 0, SEEK_SET);
//
//		if (fileLength > BufferSize()) {
//
//		}
//
//		size_t read = fread_s(array, sizeof(T) * internalSize, sizeof(T), internalSize, inFile);
//
//		fclose(inFile);
//
//		return true;
//	}
//
//	CUDA_HOST_DEVICE void clear() {
//		Memory::memset(internalArray, 0, sizeof(T) * internalSize);
//		internalSize = 0;
//	}
//
//	CUDA_HOST_DEVICE int32_t size() const { return internalSize; }
//	CUDA_HOST_DEVICE int32_t capacity() const { return internalCapacity; }
//	CUDA_HOST_DEVICE int32_t bufferSize() const { return sizeof(T) * internalCapacity; }
//
//	CUDA_HOST_DEVICE const T* begin() const { return internalArray; }
//	CUDA_HOST_DEVICE const T* end() const { return internalArray + internalSize; }
//
//	CUDA_HOST_DEVICE void print() {
//		printf("size:%d, capacity:%d\n", internalSize, internalCapacity);
//	}
//
//    CUDA_HOST_DEVICE T operator[](int32_t index) const {
//        if (index >= internalSize)
//        {
//            printf("Index out of range.\n");
//            assert(index < internalSize);
//        }
//
//        return internalArray[index];
//    }
//
//	CUDA_HOST_DEVICE T& operator[](int32_t index) {
//		if (index >= internalSize)
//		{
//			printf("Index out of range.\n");
//			assert(index < internalSize);
//		}
//
//		return internalArray[index];
//	}
//
//	CUDA_HOST_DEVICE T* data() {
//		return internalArray;
//	}
//
//private:
//	int32_t internalSize = 0;
//	const int32_t capacityIncrement = 5;
//	int32_t internalCapacity = 0;
//	T* internalArray = nullptr;
//};

template<typename T>
class Array
{
public:
    CUDA_HOST_DEVICE Array() {
        //printf("Array\n");
    }

    CUDA_HOST_DEVICE Array(int32_t inSize) {
        internalSize = inSize;
    }

    CUDA_HOST_DEVICE ~Array() {
    }

    CUDA_HOST_DEVICE void add(T value) {
        internalArray[internalSize] = value;
        internalSize++;
    }

    CUDA_HOST_DEVICE void remove(T value) {
        int32_t foundIndex = -1;

        for (auto i = 0; i < internalSize; i++) {
            if (internalArray[i] == value) {
                foundIndex = i;
                break;
            }
        }

        if (foundIndex >= 0) {
            for (auto i = foundIndex; i < internalSize - 1; i++) {
                internalArray[i] = internalArray[i + 1];
            }

            internalSize--;
        }
    }

    CUDA_HOST_DEVICE void clear() {
        Memory::memset(internalArray, 0, sizeof(T) * internalSize);
        internalSize = 0;
    }

    CUDA_HOST_DEVICE int32_t size() const { return internalSize; }
    CUDA_HOST_DEVICE int32_t capacity() const { return internalCapacity; }
    CUDA_HOST_DEVICE int32_t bufferSize() const { return sizeof(T) * internalCapacity; }

    CUDA_HOST_DEVICE const T* begin() const { return internalArray; }
    CUDA_HOST_DEVICE const T* end() const { return internalArray + internalSize; }

    CUDA_HOST_DEVICE T last() const { return internalArray[internalSize - 1]; }
    CUDA_HOST_DEVICE T& last() { return internalArray[internalSize - 1]; }

    CUDA_HOST_DEVICE void print() {
        printf("size:%d, capacity:%d\n", internalSize, internalCapacity);
    }

    CUDA_HOST_DEVICE T operator[](int32_t index) const {
        if (index >= internalSize)
        {
            printf("Index out of range.\n");
            assert(index < internalSize);
        }

        return internalArray[index];
    }

    CUDA_HOST_DEVICE T& operator[](int32_t index) {
        if (index >= internalSize)
        {
            printf("Index out of range.\n");
            assert(index < internalSize);
        }

        return internalArray[index];
    }

    CUDA_HOST_DEVICE T* data() {
        return internalArray;
    }

private:
    int32_t internalSize = 0;
    static const int32_t capacityIncrement = MAXELEMENTS;
    int32_t internalCapacity = capacityIncrement;
    T internalArray[capacityIncrement];
};