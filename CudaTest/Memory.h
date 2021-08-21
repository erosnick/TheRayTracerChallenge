#pragma once

#include "CUDA.h"

#include <cstdint>

namespace Memory {
    inline CUDA_HOST_DEVICE void* memset(void* dst, int32_t value, size_t count) {
        void* ret = dst;
        while (count--) {
            *(char*)dst = (char)value;
            dst = (char*)dst + 1; // 移动一个字节
        }
        return ret;
    }

    inline CUDA_HOST_DEVICE void* memcpy(void* dst, const void* src, int n) {
        if (dst == nullptr || src == nullptr || n <= 0)
            return nullptr;

        int* pdst = (int*)dst;
        int* psrc = (int*)src;
        char* tmp1 = nullptr;
        char* tmp2 = nullptr;
        int c1 = n / 4;
        int c2 = n % 4;

        /*if (pdst > psrc && pdst < psrc + n) 这样判断有问题*/
        if (pdst > psrc && pdst < (int*)((char*)psrc + n))
        {
            tmp1 = (char*)pdst + n - 1;
            tmp2 = (char*)psrc + n - 1;
            while (c2--)
                *tmp1-- = *tmp2--;
            /*这样有问题，忘记字节偏移
            pdst = (int *)tmp1;
            psrc = (int *)tmp2;
            */
            tmp1++; tmp2++;
            pdst = (int*)tmp1;
            psrc = (int*)tmp2;
            pdst--; psrc--;
            while (c1--)
                *pdst-- = *psrc--;
        }
        else
        {
            while (c1--)
                *pdst++ = *psrc++;
            tmp1 = (char*)pdst;
            tmp2 = (char*)psrc;
            while (c2--)
                *tmp1++ = *tmp2++;
        }

        return dst;
    }

    template<typename T>
    CUDA_HOST_DEVICE void swap(T& a, T& b) {
        auto temp = b;
        b = a;
        a = temp;
    }
}