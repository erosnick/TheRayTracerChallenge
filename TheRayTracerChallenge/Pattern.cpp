#include "Pattern.h"
#include "Shape.h"

Tuple Pattern::patternAtShape(const ShapePtr& shape, const Tuple& position) const {
    auto patternPosition = transformation.inverse() * shape->transformation.inverse() * position;
    //const auto& patternPosition = shape->transformation.inverse() * position;

    // ��ͼ����ƽ����ת�����෴�ķ�����ת
    //const auto& patternPosition = shape->transformation.inverse() * transformation * position;
    //const auto& patternPosition = transformation * shape->transformation.inverse() * position;
    //const auto& patternPosition = transformation * position;
    //const auto& patternPosition = transformation * shape->transformation * position;

    return patternAt(patternPosition);
}