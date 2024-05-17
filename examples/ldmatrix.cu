#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ROW 16
#define COL 16

__global__ void test(half* in) {
    __shared__ half smem[ROW * COL];
    const int laneid = threadIdx.x % warpSize;
    // Move data from global mem to shared mem
    const int rowi = laneid / 2;
    const int coli = (laneid % 2) * 8;
    reinterpret_cast<int4*>(&smem[rowi * COL + coli])[0] =
        reinterpret_cast<int4*>(&in[rowi * COL + coli])[0];

#if 0
    if (threadIdx.x == 0) {
        for (int i = 0; i < ROW; i++) {
            for (int j = 0; j < COL; j++) {
                printf("%f ", static_cast<float>(smem[i * COL + j]));
            }
            printf("\n");
        }
    }
#endif

    uint32_t R[4];
    half r[8];
    uint32_t smem_addr;

    {
        smem_addr = __cvta_generic_to_shared(
            &smem[(laneid % 16) * COL + (laneid / 16) * 8]);
        asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
                     : "=r"(R[0])
                     : "r"(smem_addr));
        for (int i = 0; i < 8; i++) {
            r[i] = static_cast<half>(0.0);
        }
        for (int i = 0; i < 2; i++) {
            r[i] = reinterpret_cast<half*>(&R[i / 2])[i % 2];
        }
        printf("tid:%u R: %.1f, %.1f\n", threadIdx.x, static_cast<float>(r[0]),
               static_cast<float>(r[1]));
        /*
         * Print result of above code:
         * tid:0 R:  0.0     1.0
         * tid:1 R:  2.0     3.0
         * tid:2 R:  4.0     5.0
         * tid:3 R:  6.0     7.0
         * tid:4 R:  16.0    17.0
         * tid:5 R:  18.0    19.0
         * tid:6 R:  20.0    21.0
         * tid:7 R:  22.0    23.0
         * tid:8 R:  32.0    33.0
         * tid:9 R:  34.0    35.0
         * tid:10 R: 36.0    37.0
         * tid:11 R: 38.0    39.0
         * tid:12 R: 48.0    49.0
         * tid:13 R: 50.0    51.0
         * tid:14 R: 52.0    53.0
         * tid:15 R: 54.0    55.0
         * tid:16 R: 64.0    65.0
         * tid:17 R: 66.0    67.0
         * tid:18 R: 68.0    69.0
         * tid:19 R: 70.0    71.0
         * tid:20 R: 80.0    81.0
         * tid:21 R: 82.0    83.0
         * tid:22 R: 84.0    85.0
         * tid:23 R: 86.0    87.0
         * tid:24 R: 96.0    97.0
         * tid:25 R: 98.0    99.0
         * tid:26 R: 100.0   101.0
         * tid:27 R: 102.0   103.0
         * tid:28 R: 112.0   113.0
         * tid:29 R: 114.0   115.0
         * tid:30 R: 116.0   117.0
         * tid:31 R: 118.0   119.0
         */
    }

    {
        smem_addr = __cvta_generic_to_shared(
            &smem[(laneid % 8) * COL + ((laneid / 8) % 2) * 8]);
        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(R[0]), "=r"(R[1])
            : "r"(smem_addr));
        for (int i = 0; i < 8; i++) {
            r[i] = static_cast<half>(0.0);
        }
        for (int i = 0; i < 4; i++) {
            r[i] = reinterpret_cast<half*>(&R[i / 2])[i % 2];
        }
        printf("tid:%u R: %.1f, %.1f, %.1f, %.1f\n", threadIdx.x,
               static_cast<float>(r[0]), static_cast<float>(r[1]),
               static_cast<float>(r[2]), static_cast<float>(r[3]));
        /*
         * Print result of above code:
         * tid:0 R:  0.0    1.0    8.0    9.0
         * tid:1 R:  2.0    3.0    10.0   11.0
         * tid:2 R:  4.0    5.0    12.0   13.0
         * tid:3 R:  6.0    7.0    14.0   15.0
         * tid:4 R:  16.0   17.0   24.0   25.0
         * tid:5 R:  18.0   19.0   26.0   27.0
         * tid:6 R:  20.0   21.0   28.0   29.0
         * tid:7 R:  22.0   23.0   30.0   31.0
         * tid:8 R:  32.0   33.0   40.0   41.0
         * tid:9 R:  34.0   35.0   42.0   43.0
         * tid:10 R: 36.0   37.0   44.0   45.0
         * tid:11 R: 38.0   39.0   46.0   47.0
         * tid:12 R: 48.0   49.0   56.0   57.0
         * tid:13 R: 50.0   51.0   58.0   59.0
         * tid:14 R: 52.0   53.0   60.0   61.0
         * tid:15 R: 54.0   55.0   62.0   63.0
         * tid:16 R: 64.0   65.0   72.0   73.0
         * tid:17 R: 66.0   67.0   74.0   75.0
         * tid:18 R: 68.0   69.0   76.0   77.0
         * tid:19 R: 70.0   71.0   78.0   79.0
         * tid:20 R: 80.0   81.0   88.0   89.0
         * tid:21 R: 82.0   83.0   90.0   91.0
         * tid:22 R: 84.0   85.0   92.0   93.0
         * tid:23 R: 86.0   87.0   94.0   95.0
         * tid:24 R: 96.0   97.0   104.0  105.0
         * tid:25 R: 98.0   99.0   106.0  107.0
         * tid:26 R: 100.0  101.0  108.0  109.0
         * tid:27 R: 102.0  103.0  110.0  111.0
         * tid:28 R: 112.0  113.0  120.0  121.0
         * tid:29 R: 114.0  115.0  122.0  123.0
         * tid:30 R: 116.0  117.0  124.0  125.0
         * tid:31 R: 118.0  119.0  126.0  127.0
         */
    }

    {
        smem_addr = __cvta_generic_to_shared(
            &smem[(laneid % 16) * COL + (laneid / 16) * 8]);
        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(R[0]), "=r"(R[1])
            : "r"(smem_addr));
        for (int i = 0; i < 8; i++) {
            r[i] = static_cast<half>(0.0);
        }
        for (int i = 0; i < 4; i++) {
            r[i] = reinterpret_cast<half*>(&R[i / 2])[i % 2];
        }
        printf("tid:%u R: %.1f, %.1f, %.1f, %.1f\n", threadIdx.x,
               static_cast<float>(r[0]), static_cast<float>(r[1]),
               static_cast<float>(r[2]), static_cast<float>(r[3]));
        /*
         * Print result of above code:
         * tid:0 R:  0.0    1.0    128.0  129.0
         * tid:1 R:  2.0    3.0    130.0  131.0
         * tid:2 R:  4.0    5.0    132.0  133.0
         * tid:3 R:  6.0    7.0    134.0  135.0
         * tid:4 R:  16.0   17.0   144.0  145.0
         * tid:5 R:  18.0   19.0   146.0  147.0
         * tid:6 R:  20.0   21.0   148.0  149.0
         * tid:7 R:  22.0   23.0   150.0  151.0
         * tid:8 R:  32.0   33.0   160.0  161.0
         * tid:9 R:  34.0   35.0   162.0  163.0
         * tid:10 R: 36.0   37.0   164.0  165.0
         * tid:11 R: 38.0   39.0   166.0  167.0
         * tid:12 R: 48.0   49.0   176.0  177.0
         * tid:13 R: 50.0   51.0   178.0  179.0
         * tid:14 R: 52.0   53.0   180.0  181.0
         * tid:15 R: 54.0   55.0   182.0  183.0
         * tid:16 R: 64.0   65.0   192.0  193.0
         * tid:17 R: 66.0   67.0   194.0  195.0
         * tid:18 R: 68.0   69.0   196.0  197.0
         * tid:19 R: 70.0   71.0   198.0  199.0
         * tid:20 R: 80.0   81.0   208.0  209.0
         * tid:21 R: 82.0   83.0   210.0  211.0
         * tid:22 R: 84.0   85.0   212.0  213.0
         * tid:23 R: 86.0   87.0   214.0  215.0
         * tid:24 R: 96.0   97.0   224.0  225.0
         * tid:25 R: 98.0   99.0   226.0  227.0
         * tid:26 R: 100.0  101.0  228.0  229.0
         * tid:27 R: 102.0  103.0  230.0  231.0
         * tid:28 R: 112.0  113.0  240.0  241.0
         * tid:29 R: 114.0  115.0  242.0  243.0
         * tid:30 R: 116.0  117.0  244.0  245.0
         * tid:31 R: 118.0  119.0  246.0  247.0
         */
    }

    {
        smem_addr = __cvta_generic_to_shared(
            &smem[(laneid % 16) * COL + (laneid / 16) * 8]);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
            : "r"(smem_addr));
        for (int i = 0; i < 8; i++) {
            r[i] = static_cast<half>(0.0);
        }
        for (int i = 0; i < 8; i++) {
            r[i] = reinterpret_cast<half*>(&R[i / 2])[i % 2];
        }
        printf("tid:%u R: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f\n",
               threadIdx.x, static_cast<float>(r[0]), static_cast<float>(r[1]),
               static_cast<float>(r[2]), static_cast<float>(r[3]),
               static_cast<float>(r[4]), static_cast<float>(r[5]),
               static_cast<float>(r[6]), static_cast<float>(r[7]));
        /*
         * Print result of above code:
         * tid:0 R:  0.0    1.0    128.0  129.0  8.0    9.0    136.0  137.0
         * tid:1 R:  2.0    3.0    130.0  131.0  10.0   11.0   138.0  139.0
         * tid:2 R:  4.0    5.0    132.0  133.0  12.0   13.0   140.0  141.0
         * tid:3 R:  6.0    7.0    134.0  135.0  14.0   15.0   142.0  143.0
         * tid:4 R:  16.0   17.0   144.0  145.0  24.0   25.0   152.0  153.0
         * tid:5 R:  18.0   19.0   146.0  147.0  26.0   27.0   154.0  155.0
         * tid:6 R:  20.0   21.0   148.0  149.0  28.0   29.0   156.0  157.0
         * tid:7 R:  22.0   23.0   150.0  151.0  30.0   31.0   158.0  159.0
         * tid:8 R:  32.0   33.0   160.0  161.0  40.0   41.0   168.0  169.0
         * tid:9 R:  34.0   35.0   162.0  163.0  42.0   43.0   170.0  171.0
         * tid:10 R: 36.0   37.0   164.0  165.0  44.0   45.0   172.0  173.0
         * tid:11 R: 38.0   39.0   166.0  167.0  46.0   47.0   174.0  175.0
         * tid:12 R: 48.0   49.0   176.0  177.0  56.0   57.0   184.0  185.0
         * tid:13 R: 50.0   51.0   178.0  179.0  58.0   59.0   186.0  187.0
         * tid:14 R: 52.0   53.0   180.0  181.0  60.0   61.0   188.0  189.0
         * tid:15 R: 54.0   55.0   182.0  183.0  62.0   63.0   190.0  191.0
         * tid:16 R: 64.0   65.0   192.0  193.0  72.0   73.0   200.0  201.0
         * tid:17 R: 66.0   67.0   194.0  195.0  74.0   75.0   202.0  203.0
         * tid:18 R: 68.0   69.0   196.0  197.0  76.0   77.0   204.0  205.0
         * tid:19 R: 70.0   71.0   198.0  199.0  78.0   79.0   206.0  207.0
         * tid:20 R: 80.0   81.0   208.0  209.0  88.0   89.0   216.0  217.0
         * tid:21 R: 82.0   83.0   210.0  211.0  90.0   91.0   218.0  219.0
         * tid:22 R: 84.0   85.0   212.0  213.0  92.0   93.0   220.0  221.0
         * tid:23 R: 86.0   87.0   214.0  215.0  94.0   95.0   222.0  223.0
         * tid:24 R: 96.0   97.0   224.0  225.0  104.0  105.0  232.0  233.0
         * tid:25 R: 98.0   99.0   226.0  227.0  106.0  107.0  234.0  235.0
         * tid:26 R: 100.0  101.0  228.0  229.0  108.0  109.0  236.0  237.0
         * tid:27 R: 102.0  103.0  230.0  231.0  110.0  111.0  238.0  239.0
         * tid:28 R: 112.0  113.0  240.0  241.0  120.0  121.0  248.0  249.0
         * tid:29 R: 114.0  115.0  242.0  243.0  122.0  123.0  250.0  251.0
         * tid:30 R: 116.0  117.0  244.0  245.0  124.0  125.0  252.0  253.0
         * tid:31 R: 118.0  119.0  246.0  247.0  126.0  127.0  254.0  255.0
         */
    }
}

int main() {
    half *h_in, *d_in;
    h_in = reinterpret_cast<half*>(malloc(ROW * COL * sizeof(half)));
    cudaMalloc(reinterpret_cast<void**>(&d_in), ROW * COL * sizeof(half));

    /*
     * Format of input
     *  ___ ___ ___ ___ ___ ___ ___ ___  ___ ___ ___ ___ ___ ___ ___ ___
     * |000|001|002|003|004|005|006|007||008|009|010|011|012|013|014|015|
     * |016|017|018|019|020|021|022|023||024|025|026|027|028|029|030|031|
     * |032|033|034|035|036|037|038|039||040|041|042|043|044|045|046|047|
     * |048|049|050|051|052|053|054|055||056|057|058|059|060|061|062|063|
     * |064|065|066|067|068|069|070|071||072|073|074|075|076|077|078|079|
     * |080|081|082|083|084|085|086|087||088|089|090|091|092|093|094|095|
     * |096|097|098|099|100|101|102|103||104|105|106|107|108|109|110|111|
     * |112|113|114|115|116|117|118|119||120|121|122|123|124|125|126|127|
     *  --- --- --- --- --- --- --- ---  --- --- --- --- --- --- --- ---
     * |128|129|130|131|132|133|134|135||136|137|138|139|140|141|142|143|
     * |144|145|146|147|148|149|150|151||152|153|154|155|156|157|158|159|
     * |160|161|162|163|164|165|166|167||168|169|170|171|172|173|174|175|
     * |176|177|178|179|180|181|182|183||184|185|186|187|188|189|190|191|
     * |192|193|194|195|196|197|198|199||200|201|202|203|204|205|206|207|
     * |208|209|210|211|212|213|214|215||216|217|218|219|220|221|222|223|
     * |224|225|226|227|228|229|230|231||232|233|234|235|236|237|238|239|
     * |240|241|242|243|244|245|246|247||248|249|250|251|252|253|254|255|
     */
    for (int i = 0; i < ROW * COL; i++) {
        h_in[i] = static_cast<half>(i);
    }

#if 0
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            printf("%f ", static_cast<float>(h_in[i * COL + j]));
        }
        printf("\n");
    }
#endif

    cudaMemcpy(d_in, h_in, COL * ROW * sizeof(half), cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(1);

    test<<<grid, block>>>(d_in);

    for (int i = 0; i < 16; i++) {
        printf("|");
        for (int j = 0; j < 16; j++) {
            int num = i * 16 + j;
            if (num < 10) {
                printf("0");
            }
            if (num < 100) {
                printf("0");
            }
            printf("%d|", num);
        }
        printf("\n");
    }

    free(h_in);
    cudaFree(d_in);
    return 0;
}