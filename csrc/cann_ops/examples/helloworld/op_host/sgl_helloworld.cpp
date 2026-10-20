#include "../op_kernel/sgl_helloworld_tiling.h"
#include "register/op_def_registry.h"
#include <limits>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto& x = context->GetInputShape(0)->GetStorageShape();
    const auto& y = context->GetInputShape(1)->GetStorageShape();
    if (x != y) return ge::GRAPH_FAILED;
    const int64_t size = x.GetShapeSize();
    // Eight blocks; each block copies a multiple of 32 bytes (16 halfs).
    if (size <= 0 || size % 128 != 0 ||
        size > std::numeric_limits<uint32_t>::max()) return ge::GRAPH_FAILED;
    auto* tiling = context->GetTilingData<SglHelloworldTilingData>();
    tiling->size = static_cast<uint32_t>(size);
    context->SetBlockDim(8);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SglHelloworld : public OpDef {
public:
    explicit SglHelloworld(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910_93");
    }
};
OP_ADD(SglHelloworld);
}
