cimport numpy as np
from bg_common cimport *
from libc.stdint cimport int32_t, uint32_t

cdef class MoveSequence:
    cdef public Move[4] moves
    cdef public Move[4] used_moves
    cdef public unsigned char[2] dice
    cdef public unsigned char n_moves
    cdef public unsigned char n_used_moves
    cdef public np.ndarray final_board
    cdef public bint has_final_board

    cpdef list get_moves_tuple(self)
    cpdef MoveSequence add_move(self, unsigned char src, unsigned char n)
    cpdef MoveSequence add_move_o(self, Move move)
    cdef MoveSequence copy(self)
    cdef void set_final_board(self, np.ndarray[np.int8_t, ndim=2] board)
    #def __hash__(self)
    cdef public uint32_t _toIndex(self)
    cpdef int use_move(self, unsigned char src, unsigned char n)

    @staticmethod
    #def toSequenceFromIndex(encoded)
    
    @staticmethod
    cdef MoveSequence _toSequenceFromIndex(uint32_t encoded)

cdef class MoveGenerator:
    @staticmethod
    cdef list generate_moves(np.ndarray[np.int8_t, ndim=2] board, unsigned char d1, unsigned char d2)
    
    @staticmethod
    cdef void _generate_moves_recursive(
        np.ndarray[np.int8_t, ndim=2] board,
        unsigned char move_num,
        unsigned char d1,
        unsigned char d2,
        MoveSequence curr_sequence,
        list all_sequences
    )
    
    @staticmethod
    cdef list _filter_moves(list sequences)

    @staticmethod
    cdef list generate_moves2(np.ndarray[np.int8_t, ndim=2] board, unsigned char d1, unsigned char d2 )

    @staticmethod
    cdef list generate_moves3(np.ndarray[np.int8_t, ndim=2] board, unsigned char d1, unsigned char d2 )
    
    @staticmethod
    cdef void _generate_moves_iterative(
        np.ndarray[np.int8_t, ndim=2] board,
        unsigned char d1,
        unsigned char d2,
        list all_sequences,
        unsigned char* max_moves_ptr,
        unsigned char* max_die_ptr
    )

    @staticmethod
    cdef list _filter_moves2(list sequences, unsigned char max_moves, unsigned char max_die, bint filter)


