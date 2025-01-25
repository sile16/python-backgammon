cdef bint DEBUG

cdef enum:
    BAR_POS = 0
    HOME_START_POS = 18
    HOME_END_POS = 24
    BEAR_OFF_POS = 25
    OPP_BAR_POS = 25
    OPP_BEAR_OFF_POS = 0
    ACTION_PASS = 30
    WHITE = 0
    BLACK = 1
    NONE = 2

# Optimized data structures
cdef struct Move:
    unsigned char src
    unsigned char n