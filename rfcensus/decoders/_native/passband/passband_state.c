/* passband_state.c — see passband_state.h for design notes. */

#include "passband_state.h"

int pb_process_batch(
    pb_slot_t *slots, size_t n_slots,
    const float   *energies_lin,
    const uint8_t *above_trigger,
    const uint8_t *below_release,
    const float   *db_above,
    const float   *noise_db_at_start,
    size_t n_frames,
    int64_t base_sample_offset,
    int64_t hop_samples,
    int64_t fft_size,
    int64_t frame_count,
    const pb_config_t *cfg,
    pb_event_t *events_out, size_t max_events
) {
    /* Cache config in registers for the hot loop. */
    const float   alpha            = cfg->noise_alpha;
    const int32_t trigger_frames   = cfg->trigger_frames;
    const int32_t drain_frames     = cfg->drain_frames;
    const int32_t bootstrap_frames = cfg->bootstrap_frames;

    size_t n_events = 0;
    int    overflowed = 0;

    for (size_t f = 0; f < n_frames; f++) {
        const int64_t sample_offset_at_frame_end =
            base_sample_offset + (int64_t)f * hop_samples + fft_size;

        const float   *e_row  = &energies_lin[f * n_slots];
        const uint8_t *at_row = &above_trigger[f * n_slots];
        const uint8_t *br_row = &below_release[f * n_slots];
        const float   *db_row = &db_above[f * n_slots];

        for (size_t i = 0; i < n_slots; i++) {
            pb_slot_t *s = &slots[i];
            const float energy_lin = e_row[i];

            switch ((pb_slot_state_t)s->state) {
            case PB_STATE_IDLE: {
                /* Noise EMA in linear space. First-touch initializes
                 * to the current frame's energy (matches Python's
                 * noise_floor_lin == 0.0 special case). */
                if (s->noise_floor_lin == 0.0f) {
                    s->noise_floor_lin = energy_lin;
                } else {
                    s->noise_floor_lin +=
                        alpha * (energy_lin - s->noise_floor_lin);
                }
                s->idle_frames_seen += 1;

                /* Bootstrap: don't trigger until noise estimate is
                 * trustworthy. */
                if (s->idle_frames_seen < bootstrap_frames) {
                    break;  /* out of switch, on to next slot */
                }

                if (at_row[i]) {
                    s->consec_above_trigger += 1;
                } else {
                    s->consec_above_trigger = 0;
                }

                if (s->consec_above_trigger >= trigger_frames) {
                    /* Promote to ACTIVE. */
                    s->state                 = PB_STATE_ACTIVE;
                    s->consec_above_trigger  = 0;
                    s->consec_below_release  = 0;
                    s->phase_started_frame   = frame_count;

                    if (n_events < max_events) {
                        pb_event_t *ev = &events_out[n_events++];
                        ev->kind                  = PB_EVENT_ACTIVATE;
                        ev->slot_idx              = (int32_t)i;
                        ev->sample_offset         = sample_offset_at_frame_end;
                        ev->energy_db_above_floor = db_row[i];
                        ev->noise_floor_db        = noise_db_at_start[i];
                    } else {
                        overflowed = 1;
                    }
                }
                break;
            }

            case PB_STATE_ACTIVE: {
                /* Don't update noise floor while active (signal would
                 * poison the estimate). */
                if (br_row[i]) {
                    s->state                = PB_STATE_DRAINING;
                    s->consec_below_release = 1;
                    s->phase_started_frame  = frame_count;
                }
                /* else: stay active, keep decoders running */
                break;
            }

            case PB_STATE_DRAINING:
            default: {
                if (at_row[i]) {
                    /* Energy came back — back to ACTIVE. No event;
                     * decoders are still alive. */
                    s->state                = PB_STATE_ACTIVE;
                    s->consec_below_release = 0;
                } else {
                    s->consec_below_release += 1;
                    if (s->consec_below_release >= drain_frames) {
                        /* Truly idle now. */
                        s->state                = PB_STATE_IDLE;
                        s->consec_below_release = 0;
                        s->idle_frames_seen     = 0;
                        /* Reset noise tracking — pick up the current
                         * energy as the new baseline so we don't
                         * carry stale numbers from before the
                         * transmission. */
                        s->noise_floor_lin = energy_lin;

                        if (n_events < max_events) {
                            pb_event_t *ev = &events_out[n_events++];
                            ev->kind                  = PB_EVENT_DEACTIVATE;
                            ev->slot_idx              = (int32_t)i;
                            ev->sample_offset         = sample_offset_at_frame_end;
                            ev->energy_db_above_floor = 0.0f;
                            ev->noise_floor_db        = noise_db_at_start[i];
                        } else {
                            overflowed = 1;
                        }
                    }
                }
                break;
            }
            }
        }
    }

    return overflowed ? -1 : (int)n_events;
}
