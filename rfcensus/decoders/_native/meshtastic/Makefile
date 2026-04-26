# Makefile for libmeshtastic.so — meshtastic-lite + C-API + tiny-AES-c.
#
# Targets:
#   make          — build libmeshtastic.so (default)
#   make test     — build + run unit tests
#   make clean    — remove build artifacts
#
# Configuration:
#   CC, CXX       — compilers (default: cc, c++)
#   CFLAGS_EXTRA  — extra C flags (e.g. -g for debug, -O3 for release)
#   AR            — archiver (default: ar)
#
# Output:
#   libmeshtastic.so — shared library (Linux/macOS) for ctypes consumers
#   libmeshtastic.a  — static library for embedding in other C programs
#   test_capi        — unit test binary (built by `make test`)

CC      ?= cc
CXX     ?= c++
AR      ?= ar

# Common flags. -fPIC for the shared library; -fno-exceptions / -fno-rtti
# would shrink the binary further but we want C++ exceptions available
# for std::bad_alloc handling inside the C-API allocator.
#
# v0.7.7: -O3 + -march=native for x86. AES is the biggest CPU cost
# in libmeshtastic (every received packet gets AES-CTR decrypt
# attempted against every channel); -march=native unlocks AES-NI
# on x86 which gives ~10× AES throughput. tiny-AES-c does NOT use
# AES-NI directly but the C compiler with -O3 + native intrinsics
# can vectorize the round transformations.
MESH_OPT ?= -O3 -funroll-loops
CFLAGS_BASE   = $(MESH_OPT) -Wall -Wextra -fPIC -fno-strict-aliasing
CXXFLAGS_BASE = $(MESH_OPT) -Wall -Wextra -fPIC -fno-strict-aliasing -std=c++17

UNAME_M := $(shell uname -m)
ifneq (,$(filter aarch64 arm64,$(UNAME_M)))
    CFLAGS_BASE   += -mcpu=native
    CXXFLAGS_BASE += -mcpu=native
endif
ifneq (,$(filter x86_64,$(UNAME_M)))
    CFLAGS_BASE   += -march=native
    CXXFLAGS_BASE += -march=native
endif

# MESH_MAX_CHANNELS controls the size of the channel table. Upstream
# meshtastic-lite defaults to 8 — that matches the upstream Meshtastic
# UI limit (8 channels per device). For a passive observer monitoring
# multiple presets simultaneously, we want headroom — one default
# channel per preset (9) + room for several user-defined PSKs.
# 32 is comfortable for any realistic monitoring scenario.
CHANNEL_FLAGS = -DMESH_MAX_CHANNELS=32

CFLAGS   = $(CFLAGS_BASE)   $(CHANNEL_FLAGS) -Wno-unused-parameter $(CFLAGS_EXTRA)
CXXFLAGS = $(CXXFLAGS_BASE) $(CHANNEL_FLAGS) -Wno-unused-parameter $(CFLAGS_EXTRA)

SRC_DIR   = src
THIRD_DIR = third_party

# C objects (tiny-AES-c lives in third_party/, our adapters in src/).
# The two AES blocks must be SEPARATE compilation units because
# tiny-AES-c uses compile-time #defines for key size selection.
C_OBJS = \
	$(SRC_DIR)/mesh_aes_adapter.o \
	$(SRC_DIR)/mesh_aes128_block.o \
	$(SRC_DIR)/mesh_aes256_block.o

# C++ object — just the C-API wrapper. The header-only meshtastic-lite
# lib is included via headers, no compile unit of its own.
CXX_OBJS = \
	$(SRC_DIR)/meshtastic_capi.o

ALL_OBJS = $(C_OBJS) $(CXX_OBJS)

# ─── Default target ───────────────────────────────────────────────

all: libmeshtastic.so libmeshtastic.a

libmeshtastic.so: $(ALL_OBJS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(ALL_OBJS)

libmeshtastic.a: $(ALL_OBJS)
	$(AR) rcs $@ $(ALL_OBJS)

# ─── Compilation rules ────────────────────────────────────────────

# tiny-AES-c is built FROM each block file via #include "../third_party/...":
# the block files configure the AES key size and #include aes.h. tiny-AES-c
# also has aes.c which contains the actual implementation; we need ONE copy
# of aes.c per key-size variant. To do that we let each block file's
# compilation pull in aes.h (declarations only — real impl lives in aes.c).
# That means we ALSO need to compile aes.c ONCE for each key-size variant
# — but tiny-AES-c puts the implementation in the .c file selected by the
# same macros. So the cleanest approach: include aes.c directly from each
# block file (rather than aes.h) so the compile gets the right key-size'd
# implementation. Let me update the block files accordingly.

# C compilation
$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -I$(SRC_DIR) -I$(THIRD_DIR)/tiny-AES-c -c -o $@ $<

# C++ compilation (for the capi wrapper)
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c -o $@ $<

# ─── Tests ────────────────────────────────────────────────────────

test: test_capi
	./test_capi

test_capi: tests/test_capi.cpp libmeshtastic.a
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -o $@ tests/test_capi.cpp libmeshtastic.a

# ─── Cleanup ──────────────────────────────────────────────────────

clean:
	rm -f $(ALL_OBJS) libmeshtastic.so libmeshtastic.a test_capi

.PHONY: all test clean
