#include "Pattern.h"
#include "Shape.h"

Tuple Pattern::patternAtShape(const ShapePtr& shape, const Tuple& position) const {
    auto patternPosition = transformation.inverse() * shape->transformation.inverse() * position;
    //const auto& patternPosition = shape->transformation.inverse() * position;

    // 将图案按平面旋转方向相反的方向旋转
    //const auto& patternPosition = shape->transformation.inverse() * transformation * position;
    //const auto& patternPosition = transformation * shape->transformation.inverse() * position;
    //const auto& patternPosition = transformation * position;
    //const auto& patternPosition = transformation * shape->transformation * position;

    return patternAt(patternPosition);
}