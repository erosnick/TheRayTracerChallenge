#pragma once

#include <cstdint>
#include <memory>

class Object : std::enable_shared_from_this<Object> {
public:
    Object() 
    : id(counter++) {
    }

    std::shared_ptr<Object> GetPtr() {
        return shared_from_this();
    }

    int32_t id;
    static int32_t counter;
};

inline bool operator==(const Object& a, const Object& b) {
    return (a.id == b.id);
}