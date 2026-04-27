/* passband_state.h — v0.7.15 native port of the per-slot
 * energy state machine from passband_detector.py.
 *
 * The Python implementation (`_step_state_machines_vec`) is a
 * frame-by-frame loop over n_slots that reads precomputed bool
 * matrices and updates per-slot state + emits events. With
 * 84 slots × ~250 frames per batch × ~10 batches/sec, that's
 * ~210K Python iterations/sec just for state machine work. The
 * profile shows this at 12.1% of total CPU.
 *
 * This C kernel processes the full (n_frames × n_slots) batch
 * in a single call, eliminating the Python interpreter overhead
 * for both loops. Event payloads are written into a caller-
 * provided ring; on overflow the function returns -1 and the
 * caller should resize and retry.
 *
 * No external dependencies. C99, freestanding-friendly.
 */

#ifndef PB_STATE_H
#define PB_STATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Slot state — must match Python SlotState enum ordering. */
typedef enum {
    PB_STATE_IDLE     = 0,
    PB_STATE_ACTIVE   = 1,
    PB_STATE_DRAINING = 2,
} pb_slot_state_t;

/* Event kind — must match Python event "kind" string convention. */
#define PB_EVENT_ACTIVATE   0
#define PB_EVENT_DEACTIVATE 1

/* Per-slot mutable state. ABI-stable layout: every field is
 * fixed-width, no padding ambiguity. Struct size = 32 bytes
 * (one cache line on most ARM and x86 cores can hold two of
 * these, plenty of room for 84-slot scans).
 *
 * MUST be kept in sync with the Python ctypes Structure
 * declaration in passband_state_native.py — change both at once.
 */
typedef struct {
    int32_t state;                    /* pb_slot_state_t */
    int32_t consec_above_trigger;
    int32_t consec_below_release;
    int32_t idle_frames_seen;
    int64_t phase_started_frame;
    float   noise_floor_lin;
    float   last_energy_lin;
} pb_slot_t;

/* Static (per-batch) detector parameters. */
typedef struct {
    float   noise_alpha;
    int32_t trigger_frames;
    int32_t drain_frames;
    int32_t bootstrap_frames;
} pb_config_t;

/* One emitted state-transition event. */
typedef struct {
    int32_t kind;                    /* PB_EVENT_ACTIVATE or _DEACTIVATE */
    int32_t slot_idx;                /* index into the slots array */
    int64_t sample_offset;
    float   energy_db_above_floor;   /* event payload from db_above[i] */
    float   noise_floor_db;          /* event payload from noise_db_at_start[i] */
} pb_event_t;

/* Process a full batch of frames. Iterates n_frames × n_slots,
 * mutating slots in place and writing events to events_out.
 *
 * Inputs are row-major matrices of shape (n_frames, n_slots).
 * noise_db_at_start is a single per-slot vector (snapshot at
 * batch start; matches the Python caller's convention).
 *
 * sample_offset for each frame is computed as:
 *     base_sample_offset + f * hop_samples + fft_size
 * (matches the Python `samples_consumed + f * hop + fft_size`).
 *
 * frame_count is the value of self._frame_count AT BATCH START.
 * The Python implementation captures it once per outer-loop call
 * and uses the same value for all transitions within the batch
 * (this is intentional — see the comment in _step_state_machines_vec
 * about it being batch-resolution, not frame-resolution).
 *
 * Returns the number of events written (>=0), or -1 if events_out
 * was too small (overflow). On overflow the slot state IS still
 * fully updated through the end of the batch; only the events
 * after the overflow point are dropped. Caller should size
 * events_out as 2*n_slots for safety (each slot can transition at
 * most twice per batch — once IDLE→ACTIVE, once DRAINING→IDLE).
 */
int pb_process_batch(
    pb_slot_t *slots, size_t n_slots,
    const float   *energies_lin,        /* (n_frames, n_slots) */
    const uint8_t *above_trigger,       /* (n_frames, n_slots) */
    const uint8_t *below_release,       /* (n_frames, n_slots) */
    const float   *db_above,            /* (n_frames, n_slots) */
    const float   *noise_db_at_start,   /* (n_slots,) */
    size_t n_frames,
    int64_t base_sample_offset,
    int64_t hop_samples,
    int64_t fft_size,
    int64_t frame_count,
    const pb_config_t *cfg,
    pb_event_t *events_out, size_t max_events
);

#ifdef __cplusplus
}
#endif

#endif /* PB_STATE_H */
