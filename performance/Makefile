CC=mpicc
CFLAGS=-c
LDFLAGS=-lm
EXEC=peval
SRCS=peval.c
OBJS=$(SRCS:.c=.o)
$(EXEC):$(OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)
.c.o:
	$(CC) $(CFLAGS) $< -o $@
