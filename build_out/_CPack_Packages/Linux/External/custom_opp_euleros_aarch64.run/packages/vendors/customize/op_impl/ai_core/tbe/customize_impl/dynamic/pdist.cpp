#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;    

template <uint32_t N>
struct Array
{
    uint32_t data[N];
    template <typename... Args>
    __aicore__ Array(Args... args) : data{uint32_t(args)...} {}
};

template <typename T, uint32_t N = 1>
struct Tensor
{
    using GM = AscendC::GlobalTensor<T>;
    Array<N> dim;
    GM &gm;

    __aicore__ Tensor(GM &gm) : gm(gm), dim(Array<1>(size())) {}
    __aicore__ Tensor(GM &gm, const Array<N> &dim) : gm(gm), dim(dim) {}

    __aicore__ uint32_t size() const { return gm.GetSize(); }

    __aicore__ uint32_t dim_size() const { return N; }

    __aicore__ __gm__ T &operator[](int index) { return (gm(index)); }

    __aicore__ __gm__ T &operator[](const Array<N> &indexs)
    {
        int index = 0;
        for (int i = N - 1, p = 1; i >= 0; i--)
        {
            index += indexs.data[i] * p;
            p *= dim.data[i];
        }
        return gm(index);
    }
};

template<typename TYPE_X>
class Pdist
{
public:
    __aicore__ inline Pdist() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t n,
        uint32_t m,
        float p,
        uint32_t aivNum)
    {
        this->n = n;
        this->m = m;
        this->p = p;
        this->aivNum=aivNum;

        uint32_t size = n * m;
        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x, n * m);
        yGm.SetGlobalBuffer((__gm__ TYPE_X *)y, (n * (n - 1) / 2 )/64*64+64 );

        this->tileLength= m / 64 * 64 + 64;

        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));

        uint32_t mm = (m * sizeof(TYPE_X) + 32 - 1) / 32 * 32 / sizeof(TYPE_X);

        this->mmm = m / 64 * 64 + 64;
        uint32_t inner = (m * sizeof(float) + 32 - 1) / 32 * 32 / sizeof(float);

        pipe.InitBuffer(tmp1, mmm * sizeof(TYPE_X));
        pipe.InitBuffer(tmp2, mmm * sizeof(TYPE_X));
        pipe.InitBuffer(tmp3, mmm * sizeof(float));
        pipe.InitBuffer(tmp4, mmm * sizeof(float));
        pipe.InitBuffer(tmp5, mmm * sizeof(float));
        pipe.InitBuffer(tmp6, 128 * sizeof(int));
        
        // printf("%d %d %d\n",GetBlockNum(), n, m);
        // printf("%d\n", GetBlockIdx());
        this->loopCounts = n / aivNum;
        this->henum=GetBlockIdx();
    } 




    // __aicore__ inline void Process()
    // {
    //     Tensor<TYPE_X, 2> x(xGm, {n, m});
    //     Tensor<TYPE_X, 1> y(yGm, {n * (n - 1) / 2});

    //     auto p1 = tmp1.Get<TYPE_X>();
    //     auto p2 = tmp2.Get<TYPE_X>();
    //     auto p3 = tmp3.Get<TYPE_X>();
    //     auto p4 = tmp4.Get<float>();
    //     auto psum = tmp5.Get<float>();

    //     int idx = 0;
    //     for (int i = 0; i < n; i++) {
    //         for (int j = i + 1; j < n; j++) {
                
    //             uint32_t mmm = m / 64 * 64 + 64;
    //             // auto p5 = tmp5.Get<float>();
    //             DataCopy(p1, xGm[i * m], mmm);
    //             DataCopy(p2, xGm[j * m], mmm);
    //             // for (int k = 0; k < m; k++) {
    //             //     p1(k) = xGm(i * m + k);
    //             //     p2(k) = xGm(j * m + k);
    //             //     printf("%d %d %f %f\n", i, j, p1(k), p2(k));
    //             // }
    //             p1(0), p2(0);
    //             Sub(p3, p1, p2, mmm);
    //             if constexpr (std::is_same_v<TYPE_X, float>) {
    //                 Power(p3, p3, p, mmm);
    //                 uint32_t inner = (m * sizeof(float) + 32 - 1) / 32 * 32 / sizeof(float);
    //                 SumParams sp{1, inner, m};
    //                 Sum(psum, p3, sp);
    //                 p3(0) = psum(0);
    //                 Power(p3, p3, 1.0f / p, 1);
    //                 y[idx++] = p3(0);
    //             } else {
    //                 Cast(p4, p3, RoundMode::CAST_NONE, mmm);
    //                 Power(p4, p4, p, mmm);
    //                 uint32_t inner = (m * sizeof(float) + 32 - 1) / 32 * 32 / sizeof(float);
    //                 SumParams sp{1, inner, m};
    //                 Sum(psum, p4, sp);
    //                 p4(0) = psum(0);
    //                 Power(p4, p4, 1.0f / p, 1);
    //                 y[idx++] = p4(0);
    //             }
    //         }
    //     }
    // }


#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

    
    __aicore__ inline void Process2() {
        
        uint32_t thisloopCount = loopCounts;

        if (this->henum + 1 == this->aivNum)
        {
           thisloopCount = n % aivNum + thisloopCount;
        }
        
        for (uint32_t i = 0; i < thisloopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i);
        }

    }

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t length) {

        //printf("ComyIn\n");

        LocalTensor<TYPE_X> xx = Q_x.AllocTensor<TYPE_X>();



        DataCopy(xx,xGm[(loopCounts*(this->henum)+progress) * m], length);

        Q_x.EnQue(xx);

        //printf("%d\n",progress);
    }
    __aicore__ inline void Compute(uint32_t progress, uint32_t length) {

        LocalTensor<TYPE_X> xx = Q_x.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> yy = Q_y.AllocTensor<TYPE_X>();

        Tensor<TYPE_X, 2> x(xGm, {n, m});
        Tensor<TYPE_X, 1> y(yGm, {n * (n - 1) / 2});

        uint32_t idx=loopCounts*(this->henum)+progress;

        auto p2 = tmp2.Get<TYPE_X>();
        auto p3 = tmp3.Get<TYPE_X>();
        auto p4 = tmp4.Get<float>();
        auto psum = tmp5.Get<float>();

        uint32_t cnt=0;

        for (uint32_t j = idx + 1; j < n; j++) {
            
            DataCopy(p2, xGm[j * m], mmm);

            xx(0), p2(0);

        // if(henum==0&&progress==1)
        // {
        //     for(int ii=0;ii<m;ii++)
        //     {
        //         printf("xx=%f %f\n",xx(ii),x[{idx,ii}]);
        //     }
        //     printf("\n");


        //     for(int ii=0;ii<m;ii++)
        //     {
        //         printf("p2=%f %f\n",p2(ii),x[{j,ii}]);
        //     }
        //     printf("\n");

        // }

            Sub(p3, xx, p2, mmm);
            if constexpr (std::is_same_v<TYPE_X, float>) {
                Power(p3, p3, p, mmm);
                uint32_t inner = (m * sizeof(float) + 32 - 1) / 32 * 32 / sizeof(float);
                SumParams sp{1, inner, m};
                Sum(psum, p3, sp);
                p3(0) = psum(0);
                Power(p3, p3, 1.0f / p, 1);
                yy(cnt++) = p3(0);
            } else {
                Cast(p4, p3, RoundMode::CAST_NONE, mmm);
                Power(p4, p4, p, mmm);
                uint32_t inner = (m * sizeof(float) + 32 - 1) / 32 * 32 / sizeof(float);
                SumParams sp{1, inner, m};
                Sum(psum, p4, sp);
                p4(0) = psum(0);
                Power(p4, p4, 1.0f / p, 1);
                yy(cnt++) = p4(0);
            }
        }

        // if(henum==0&&progress==1)
        // {
        //     for(int ii=0;ii<cnt;ii++)
        //     {
        //         printf("%f\n",yy(ii));

        //     }
        // }

        Q_x.FreeTensor(xx);
        Q_y.EnQue<TYPE_X>(yy);

        //printf("ComputeEND\n");
    }
    __aicore__ inline void CopyOut(uint32_t progress) {

        //printf("CopyOut\n");
        
        LocalTensor<TYPE_X> yy = Q_y.DeQue<TYPE_X>();
        uint32_t idx=loopCounts*(this->henum)+progress+1;
        uint32_t position=((2*n-idx)*(idx-1))/2;

        //printf("position:");
        //printf("%d %d %d %d %d\n",progress, loopCounts, this->henum, loopCounts*(this->henum)+progress+1, position);
        

        for(uint32_t i=0;i<n-idx;i++)
          {
              yGm(position+i)=yy(i);
          }

          Q_y.FreeTensor(yy);

        //printf("CopyOutEND\n");

    }


private:
    TPipe pipe;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<float> fxGm;
    AscendC::GlobalTensor<TYPE_X> yGm;
    TBuf<QuePosition::VECCALC> tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;

    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;

    TQue<QuePosition::VECOUT, 1> workQueue;

    uint32_t tileLength, tileNum, loopCounts, mmm;
    uint32_t n, m;
    uint32_t aivNum;
    uint32_t henum;
    float p;
};

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    Pdist<DTYPE_X> op;
    op.Init(
        x, y,
        tiling_data.n,
        tiling_data.m,
        tiling_data.p,
        tiling_data.aivNum
        );
    op.Process2();
}