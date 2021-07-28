#pragma once

#include <cstdint>

class Object {
public:
    Object() 
    : id(counter++) {
    }

    int32_t id;
    static int32_t counter;
};

inline bool operator==(const Object& a, const Object& b) {
    return (a.id == b.id);
}