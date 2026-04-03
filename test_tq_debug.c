/* test_tq_debug.c — debug the dequant issue */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TQ_IMPLEMENTATION
#include "include/tq.h"

int main(void) {
    const char *path = "/tmp/test_debug.tq";
    FILE *fp;
    tq_header_t hdr;
    tq_tensor_t desc;
    uint8_t packed[4] = {0x24, 0x24, 0x24, 0x24};
    long pos, aligned;
    tq_file_t f;
    float dst[16];
    
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = TQ_MAGIC;
    hdr.version = TQ_VERSION;
    hdr.tensor_count = 1;
    hdr.model_family_id = TQ_FAMILY_UNKNOWN;
    
    memset(&desc, 0, sizeof(desc));
    strncpy(desc.name, "weight", sizeof(desc.name) - 1);
    desc.b = 2;
    desc.rows = 4;
    desc.cols = 4;
    desc.wht_seed = 0;
    desc.frame_offset = 0;
    desc.frame_size = 0;
    desc.unpacked_size = 4;
    
    pos = (long)(sizeof(tq_header_t) + sizeof(tq_tensor_t));
    aligned = (pos + 63) & ~63L;
    hdr.data_offset = (uint64_t)aligned;
    hdr.total_data_size = 4;
    
    fp = fopen(path, "wb");
    if (!fp) { printf("Failed to open file\n"); return 1; }
    
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(&desc, sizeof(desc), 1, fp);
    
    pos = ftell(fp);
    while (pos < aligned) { uint8_t zero = 0; fwrite(&zero, 1, 1, fp); pos++; }
    
    fwrite(packed, 1, 4, fp);
    fclose(fp);
    
    if (tq_mmap(path, &f) != 0) { printf("tq_mmap failed\n"); return 1; }
    
    printf("b=%u, wht_seed=%llu\n", f.tensors[0].b, (unsigned long long)f.tensors[0].wht_seed);
    
    memset(dst, 0, sizeof(dst));
    tq_dequant(&f, 0, dst);
    
    printf("Dequantized values:\n");
    for (int i = 0; i < 4; i++) {
        printf("  dst[%d] = %f\n", i, dst[i]);
    }
    
    int ok = (dst[0] == -1.0f && dst[1] > -0.35f && dst[1] < -0.30f &&
              dst[2] > 0.30f && dst[2] < 0.35f && dst[3] == -1.0f);
    printf("Test %s\n", ok ? "PASSED" : "FAILED");
    
    tq_munmap(&f);
    remove(path);
    return ok ? 0 : 1;
}
