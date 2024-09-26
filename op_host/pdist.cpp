
#include "pdist_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    const uint32_t BLOCK_SIZE = 32;

  PdistTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  //context->SetBlockDim(8);

  uint32_t n = x1_shape->GetStorageShape().GetDim(0);
  uint32_t m = x1_shape->GetStorageShape().GetDim(1);
  const auto attrs = context->GetAttrs();
  const float* p = attrs->GetFloat(0);
  tiling.set_n(n);
  tiling.set_m(m);
  tiling.set_p(*p);

  


  int32_t NUM = 6;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size; ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();//获取当前硬件平台的核数

    uint32_t total_length = 0, min_length = context->GetInputTensor(0)->GetShapeSize();

    for (int i = 0; i < 1; ++i) {
        total_length = std::max<uint32_t>(total_length, context->GetInputTensor(i)->GetShapeSize());//获取输入数据的最大长度
        min_length = std::min<uint32_t>(min_length, context->GetInputTensor(i)->GetShapeSize());//获取输入数据的最小长度
    }
    uint32_t input_data_length = context->GetInputTensor(0)->GetShapeSize();

    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t sizeofdatatype;
    if (dt == ge::DT_FLOAT16) {
        sizeofdatatype = 2;
    }
    else{
        sizeofdatatype = 4;
    }


    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;//一个block的数据量
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;//每个核的block数
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;//保证每个核的block数是8的倍数

    uint32_t block_size = tiling_size * ALIGN_NUM;//每次要处理的数据量
    if (total_length != min_length) {
        block_size = std::min(block_size, min_length);
        while (min_length % block_size || min_length % ALIGN_NUM) {
            block_size -= 1;
        }
    }

    aivNum = (aivNum < total_length / block_size) ? aivNum : (total_length / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;

    uint32_t core_size = (total_length / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);//每个核的数据量
    uint32_t core_remain = total_length - aivNum * core_size;


    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);
    tiling.set_total_length(total_length);


    //!!!!
    context->SetBlockDim(aivNum);




  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Pdist : public OpDef {
public:
    explicit Pdist(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("p").AttrType(OPTIONAL).Float(2.0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Pdist);
}
