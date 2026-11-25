#ifndef RPROFREP_CUDA_VALUE_H
#define RPROFREP_CUDA_VALUE_H

/*
 * Generic msgpack encoding of a CUPTI parameter structure field.
 *
 * The field types of the CUPTI parameter structures change across CUDA
 * releases, so the generated writers (cuda_rt_args.h / cuda_drv_args.h) do not
 * hard-code them: the encoding is selected at compile time from the C type
 * class of the field (__builtin_classify_type, GCC/Clang):
 *   pointers            -> uint (address)
 *   integers / enums    -> int or uint (according to signedness)
 *   float / double      -> double
 *   dim3                -> [x, y, z]
 *   other structures    -> nil
 */

#include <stdint.h>
#include <string.h>
#include <vector_types.h>
#include "msgpack.h"

#define RP_ALIGN8(x) (((x) + 7u) & ~7u)

enum {
    RP_TC_VOID = 0, RP_TC_INTEGER = 1, RP_TC_CHAR = 2, RP_TC_ENUM = 3, RP_TC_BOOL = 4,
    RP_TC_POINTER = 5, RP_TC_REFERENCE = 6, RP_TC_OFFSET = 7, RP_TC_REAL = 8,
    RP_TC_COMPLEX = 9, RP_TC_FUNCTION = 10, RP_TC_METHOD = 11, RP_TC_RECORD = 12,
    RP_TC_UNION = 13, RP_TC_ARRAY = 14
};

static inline void rp_enc_value(msgpack_buffer_t* buf, const void* p, size_t sz, int cls, int is_signed, int is_dim3)
{
    switch (cls) {
        case RP_TC_POINTER: {
            uintptr_t v = 0;
            memcpy(&v, p, sizeof(v) < sz ? sizeof(v) : sz);
            msgpack_encode_uint(buf, (uint64_t)v);
            return;
        }
        case RP_TC_REAL: {
            if (sz == sizeof(float))       { float f;  memcpy(&f, p, sizeof(f)); msgpack_encode_double(buf, (double)f); }
            else if (sz == sizeof(double)) { double d; memcpy(&d, p, sizeof(d)); msgpack_encode_double(buf, d); }
            else msgpack_encode_nil(buf);
            return;
        }
        case RP_TC_INTEGER: case RP_TC_CHAR: case RP_TC_ENUM: case RP_TC_BOOL: {
            switch (sz) {
                case 1: { uint8_t v;  memcpy(&v, p, 1); if (is_signed) msgpack_encode_int(buf, (int8_t)v);  else msgpack_encode_uint(buf, v); return; }
                case 2: { uint16_t v; memcpy(&v, p, 2); if (is_signed) msgpack_encode_int(buf, (int16_t)v); else msgpack_encode_uint(buf, v); return; }
                case 4: { uint32_t v; memcpy(&v, p, 4); if (is_signed) msgpack_encode_int(buf, (int32_t)v); else msgpack_encode_uint(buf, v); return; }
                case 8: { uint64_t v; memcpy(&v, p, 8); if (is_signed) msgpack_encode_int(buf, (int64_t)v); else msgpack_encode_uint(buf, v); return; }
                default: msgpack_encode_nil(buf); return;
            }
        }
        case RP_TC_RECORD:
            if (is_dim3) {
                dim3 d; memcpy(&d, p, sizeof(d));
                msgpack_encode_array(buf, 3);
                msgpack_encode_uint(buf, d.x);
                msgpack_encode_uint(buf, d.y);
                msgpack_encode_uint(buf, d.z);
                return;
            }
            msgpack_encode_nil(buf);
            return;
        default:
            msgpack_encode_nil(buf);
            return;
    }
}

#define RP_IS_SIGNED(x) _Generic((x),                     \
    signed char: 1, short: 1, int: 1, long: 1, long long: 1, \
    char: ((char)-1 < 0),                                   \
    default: 0)

#define RP_IS_DIM3(x) _Generic((x), dim3: 1, default: 0)

#define RP_ENC_VALUE(buf, x) \
    rp_enc_value((buf), (const void*)&(x), sizeof(x), __builtin_classify_type(x), RP_IS_SIGNED(x), RP_IS_DIM3(x))

#endif // RPROFREP_CUDA_VALUE_H
