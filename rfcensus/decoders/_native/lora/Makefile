# Build the LoRa decoder as a shared lib (for Python ctypes loading)
# AND a static lib (for direct C linking from a test harness).
#
# v0.7.7: bumped from -O2 to -O3 -march=native -ffast-math
# -funroll-loops on x86. Modern GCC 11+ on x86-64 with SSE4.2/AVX2
# vectorizes the FFT butterflies cleanly, ~25-35% throughput gain
# on the LoRa demod hot path.
#
# v0.7.11: removed -ffast-math from default LORA_OPT after observing
# AVX-promotion crashes ("Illegal instruction") in some hypervisor
# environments. The interaction of -ffast-math + -march=native +
# our hand-written SSE3 dechirp intrinsics caused GCC to emit AVX
# instructions in surrounding code that triggered SIGILL on certain
# VMs. The pffft FFT backend has its own internal SIMD that doesn't
# need our -ffast-math, and the channelizer + blind probe gains in
# v0.7.11 more than offset losing whatever -ffast-math was buying
# for us before. End-to-end recall preserved (9/9 packets on the
# real capture), keep-up with v0.7.11 features ON is still ~14-18%.
# To re-enable for benchmarking on a clean (non-VM) build:
#   make LORA_OPT='-O3 -ffast-math -funroll-loops'
#
# Pi 5 (Cortex-A76) handles -O3 fine; if you're building on a Pi 4
# (A72) or older ARM and see regressions, set `LORA_OPT=-O2`.

CC      ?= cc
LORA_OPT ?= -O3 -funroll-loops
CFLAGS  ?= $(LORA_OPT) -std=c99 -fPIC -Wall -Wextra -Wno-unused-parameter \
           -Wno-unused-variable -Wno-unused-but-set-variable \
           -fno-strict-aliasing
LDFLAGS ?=

# Detect platform for NEON / SSE flags. -march=native (x86) and
# -mcpu=native (ARM) unlock the build host's full instruction set:
# AVX2/AVX-512/AES-NI on Intel, NEON/FP16/dotprod on Cortex-A76
# (Pi 5), NEON only on A72 (Pi 4) / A53 (Pi Zero 2W).
#
# CROSS-COMPILE / FLEET BUILDS: if you're building on machine A
# for deployment to machine B with a different CPU, override
# -mcpu/-march on the make line:
#   Pi 4 build:  make CFLAGS_EXTRA="-mcpu=cortex-a72"
#   Pi 5 build:  make CFLAGS_EXTRA="-mcpu=cortex-a76"
#   Pi 0 2W:     make CFLAGS_EXTRA="-mcpu=cortex-a53"
#
# All of these are aarch64 (we don't ship an armv7 32-bit build).
# GCC -O3 + -mcpu=<arch> auto-vectorizes the FFT butterflies and
# the magnitude-squared loops in the LoRa demod into NEON 2x32-bit
# float ops, which is where the bulk of the speedup comes from.
UNAME_M := $(shell uname -m)
ifneq (,$(filter aarch64 arm64,$(UNAME_M)))
    CFLAGS += -mcpu=native
endif
ifneq (,$(filter x86_64,$(UNAME_M)))
    CFLAGS += -march=native
endif

# v0.7.9: pluggable FFT backend. Default = pffft (Pretty Fast FFT
# by Pommier). Falls back to kiss_fft on `LORA_FFT_BACKEND=kiss`
# for portability or numerical-comparison testing.
#
# pffft uses SSE on x86 and NEON on aarch64, giving 2-3× FFT speedup
# vs kiss_fft on Pi 5 (Cortex-A76). End-to-end LoRa decoder speedup
# is 30-50% since FFT dominates the chirp-detect / dechirp / SNR
# paths. On x86_64 with AVX2 the gap is smaller (~1.5-2×) because
# GCC -O3 partially auto-vectorizes kiss_fft.
LORA_FFT_BACKEND ?= pffft

ifeq ($(LORA_FFT_BACKEND),pffft)
    FFT_SRCS = lora_fft_pffft.c pffft.c
    # pffft uses _mm_malloc / posix_memalign internally and needs
    # alignment guarantees; -fno-strict-aliasing is already on which
    # is required for the cf_t / lora_fft_cpx_t casting we do.
else ifeq ($(LORA_FFT_BACKEND),kiss)
    FFT_SRCS = lora_fft_kiss.c kiss_fft.c
else
    $(error Unknown LORA_FFT_BACKEND=$(LORA_FFT_BACKEND), expected 'pffft' or 'kiss')
endif

SRCS = lora_demod.c lora_chirp.c lora_codec.c lora_probe.c lora_channelizer.c $(FFT_SRCS)
OBJS = $(SRCS:.c=.o)

LIB_NAME    = liblora_demod.so
STATIC_NAME = liblora_demod.a

.PHONY: all clean test

all: $(LIB_NAME) $(STATIC_NAME)

$(LIB_NAME): $(OBJS)
	$(CC) $(CFLAGS) -shared -o $@ $(OBJS) -lm $(LDFLAGS)

$(STATIC_NAME): $(OBJS)
	ar rcs $@ $(OBJS)

# Per-object rules. lora_*.c sources depend on the project headers;
# the FFT backend libraries (pffft.c, kiss_fft.c) are self-contained
# and only depend on their own headers.
lora_demod.o: lora_demod.c lora_internal.h lora_demod.h lora_fft.h
	$(CC) $(CFLAGS) -c -o $@ $<
lora_chirp.o: lora_chirp.c lora_internal.h lora_demod.h lora_fft.h
	$(CC) $(CFLAGS) -c -o $@ $<
lora_codec.o: lora_codec.c lora_internal.h lora_demod.h
	$(CC) $(CFLAGS) -c -o $@ $<
lora_fft_pffft.o: lora_fft_pffft.c lora_fft.h pffft.h
	$(CC) $(CFLAGS) -c -o $@ $<
lora_fft_kiss.o: lora_fft_kiss.c lora_fft.h kiss_fft.h
	$(CC) $(CFLAGS) -c -o $@ $<
pffft.o: pffft.c pffft.h
	$(CC) $(CFLAGS) -D_GNU_SOURCE -c -o $@ pffft.c
kiss_fft.o: kiss_fft.c kiss_fft.h
	$(CC) $(CFLAGS) -c -o $@ $<
lora_probe.o: lora_probe.c lora_probe.h lora_fft.h
	$(CC) $(CFLAGS) -c -o $@ lora_probe.c
lora_channelizer.o: lora_channelizer.c lora_channelizer.h
	$(CC) $(CFLAGS) -c -o $@ lora_channelizer.c

# Test harness — reads a .cu8 capture and dumps any decoded packets.
test_decode: test_decode.c $(STATIC_NAME)
	$(CC) $(CFLAGS) -o $@ test_decode.c $(STATIC_NAME) -lm

clean:
	rm -f $(OBJS) $(LIB_NAME) $(STATIC_NAME) test_decode
