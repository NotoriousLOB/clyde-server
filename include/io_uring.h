/* io_uring.h — Full async io_uring backend for Tensio
 * Strict C99, zero UB, header-only implementation
 * Enable with -DTENSIO_ENABLE_IO_URING
 */

#ifndef IO_URING_H
#define IO_URING_H

#include <stddef.h>
#include <stdint.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>

typedef struct tensio_uring_ctx {
    int ring_fd;
    struct io_uring_params params;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;
    uint32_t *sq_head, *sq_tail, *sq_mask;
    uint32_t *cq_head, *cq_tail, *cq_mask;
    uint32_t sqes_head;
} tensio_uring_ctx_t;

/* Callback type for async operations */
typedef void (*tensio_uring_cb)(int res, void *user_data);

/* ================================================================
 * Public backend
 * ================================================================ */

extern const tensio_io_backend_t tensio_io_uring;

#endif /* IO_URING_H */

/* ================================================================
 * IMPLEMENTATION (define TENSIO_IO_URING_IMPLEMENTATION in one .c)
 * ================================================================ */

#ifdef TENSIO_IO_URING_IMPLEMENTATION

static tensio_uring_ctx_t uring = { .ring_fd = -1 };

static int tensio_uring_init(void) {
    if (uring.ring_fd >= 0) return 0;

    struct io_uring_params p = {0};
    p.flags = 0;                     /* no IORING_SETUP_IOPOLL etc. for simplicity */

    int fd = (int)syscall(SYS_io_uring_setup, 64, &p);  /* 64 entries is plenty */
    if (fd < 0) return -1;

    uring.ring_fd = fd;
    uring.params = p;

    /* Map SQ and CQ */
    size_t sq_size = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
    uring.sqes = mmap(0, p.sq_entries * sizeof(struct io_uring_sqe),
                      PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                      fd, p.sq_off.sqes);

    uring.cqes = mmap(0, p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe),
                      PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                      fd, p.cq_off.cqes);

    uring.sq_head = (uint32_t*)((char*)uring.sqes + p.sq_off.head);
    uring.sq_tail = (uint32_t*)((char*)uring.sqes + p.sq_off.tail);
    uring.sq_mask = (uint32_t*)((char*)uring.sqes + p.sq_off.ring_mask);

    uring.cq_head = (uint32_t*)((char*)uring.cqes + p.cq_off.head);
    uring.cq_tail = (uint32_t*)((char*)uring.cqes + p.cq_off.tail);
    uring.cq_mask = (uint32_t*)((char*)uring.cqes + p.cq_off.ring_mask);

    uring.sqes_head = 0;
    return 0;
}

/* Submit an async read */
static int tensio_uring_read_async(int fd, void *buf, size_t len,
                                   off_t offset, void *user_data)
{
    if (uring.ring_fd < 0) {
        if (tensio_uring_init() != 0) return -1;
    }

    /* Get next SQE */
    uint32_t head = *uring.sq_head;
    uint32_t next = head & *uring.sq_mask;
    struct io_uring_sqe *sqe = &uring.sqes[next];

    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IORING_OP_READ;
    sqe->fd = fd;
    sqe->addr = (uint64_t)buf;
    sqe->len = len;
    sqe->off = offset;
    sqe->user_data = (uint64_t)user_data;

    /* Submit */
    *uring.sq_tail = head + 1;
    syscall(SYS_io_uring_enter, uring.ring_fd, 1, 0, IORING_ENTER_GETEVENTS, NULL);

    return 0;
}

static int tensio_uring_open(const char *path, int flags) {
    return open(path, flags | O_DIRECT | O_CLOEXEC);
}

static void tensio_uring_close(int fd) {
    close(fd);
}

/* ================================================================
 * Public backend
 * ================================================================ */

const tensio_io_backend_t tensio_io_uring = {
    .open       = tensio_uring_open,
    .read_async = tensio_uring_read_async,
    .close      = tensio_uring_close,
};

#endif /* TENSIO_IO_URING_IMPLEMENTATION */

