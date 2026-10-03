CC      = cc
CFLAGS  = -Wall -Wextra -Werror -pedantic -Isrc
LDFLAGS = -lm

LIB     = libml.a
TRAIN   = train
PREDICT = predict
PRECISION = precision

LIB_SRCS = src/dataset_load.c \
           src/dataset_free.c \
           src/linreg_predict.c \
           src/squared_error_cost.c \
           src/compute_gradient.c \
           src/gradient_descent.c \
           src/weight_and_bias_save.c \
           src/weight_and_bias_load.c \
           src/z_score_normalisation.c \
           src/weight_and_bias_denormalize.c \
           src/r_squared.c

LIB_OBJS = $(LIB_SRCS:.c=.o)

all: $(TRAIN) $(PREDICT) $(PRECISION)

# Bundle the library objects into a static archive.
$(LIB): $(LIB_OBJS)
	ar rcs $@ $^

$(TRAIN): train.o $(LIB)
	$(CC) $(CFLAGS) train.o $(LIB) $(LDFLAGS) -o $@

$(PREDICT): predict.o $(LIB)
	$(CC) $(CFLAGS) predict.o $(LIB) $(LDFLAGS) -o $@

# Every .o depends on the header, so editing it rebuilds everything.
$(PRECISION): precision.o $(LIB)
	$(CC) $(CFLAGS) precision.o $(LIB) $(LDFLAGS) -o $@

%.o: %.c src/ml_operations.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(LIB_OBJS) train.o predict.o precision.o

fclean: clean
	rm -f $(LIB) $(TRAIN) $(PREDICT) $(PRECISION)

re: fclean all

.PHONY: all clean fclean re
