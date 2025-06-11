// ===== AADTypes.h =====
#pragma once

enum class AADOpType {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    LOG = 4,
    EXP = 5,
    SQRT = 6,
    ERF = 7,
    NORM_CDF = 8,
    NEG = 9
};

struct GPUTapeEntry {
    int result_idx;
    int op_type;
    int input1_idx;
    int input2_idx;
    double constant;
    double partial1;
    double partial2;
};



