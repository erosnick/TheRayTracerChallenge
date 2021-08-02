#include "Pattern.h"
#include "Shape.h"

Tuple Pattern::patternAtShape(const ShapePtr& shape, const Tuple& position) const {
    //auto patternPosition = transformation.inverse() * shape->transformation.inverse() * position;
    const auto& patternPosition = position;

    return patternAt(patternPosition);
}