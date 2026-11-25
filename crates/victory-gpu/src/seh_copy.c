#include <stddef.h>
#include <string.h>
#include <windows.h>

/*
 * Fast in-process memory copy guarded by Windows SEH (__try/__except).
 *
 * Used to read potentially-unmapped title memory into writable host shadow
 * buffers at full native memcpy bandwidth (~30+ GB/s) instead of paying
 * ReadProcessMemory kernel transition + bounce overhead (~5 GB/s).
 *
 * Only catches read/page faults (EXCEPTION_ACCESS_VIOLATION /
 * EXCEPTION_IN_PAGE_ERROR); returns 1 on complete success and 0 if any page
 * faults so the caller can fall back to read_sparse / read_pieces.
 */
int victory_seh_copy_read(void* dst, const void* src, size_t len) {
    if (len == 0) {
        return 1;
    }
    __try {
        memcpy(dst, src, len);
        return 1;
    } __except (
        GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ||
        GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR
            ? EXCEPTION_EXECUTE_HANDLER
            : EXCEPTION_CONTINUE_SEARCH
    ) {
        return 0;
    }
}
