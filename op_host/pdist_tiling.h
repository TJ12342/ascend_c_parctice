
#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(PdistTilingData)
  TILING_DATA_FIELD_DEF(float, p);
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(uint32_t, total_length);
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(Pdist, PdistTilingData)
}
