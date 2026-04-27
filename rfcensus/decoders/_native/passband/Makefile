# Build the passband state machine as a small shared lib for ctypes.
#
# v0.7.15: ports the per-slot state machine from
# passband_detector.py to C. ~12% of total CPU per the v0.7.14
# profile; mostly Python interpreter overhead in the per-slot
# loop. The C kernel processes the full (n_frames × n_slots)
# batch in one call.
#
# Self-contained — no LoRa or numpy deps. Tiny .so (~10 KB).

CC      ?= cc
PB_OPT  ?= -O3 -funroll-loops
CFLAGS  ?= $(PB_OPT) -std=c99 -fPIC -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?=

# CPU-specific tuning (mirrors lora/Makefile conventions).
UNAME_M := $(shell uname -m)
ifneq (,$(filter aarch64 arm64,$(UNAME_M)))
    CFLAGS += -mcpu=native
endif
ifneq (,$(filter x86_64,$(UNAME_M)))
    CFLAGS += -march=native
endif

SRCS = passband_state.c
OBJS = $(SRCS:.c=.o)

LIB_NAME    = libpassband_state.so
STATIC_NAME = libpassband_state.a

.PHONY: all clean

all: $(LIB_NAME) $(STATIC_NAME)

$(LIB_NAME): $(OBJS)
	$(CC) $(CFLAGS) -shared -o $@ $(OBJS) $(LDFLAGS)

$(STATIC_NAME): $(OBJS)
	ar rcs $@ $(OBJS)

passband_state.o: passband_state.c passband_state.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(LIB_NAME) $(STATIC_NAME)
