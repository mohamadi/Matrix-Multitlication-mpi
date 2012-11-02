CC=mpicc
CFLAGS=-c
LDFLAGS=-lm
EXEC=muly
SRCS=mult.c
OBJS=$(SRCS:.c=.o)
$(EXEC):$(OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)
.c.o:
	$(CC) $(CFLAGS) $< -o $@
