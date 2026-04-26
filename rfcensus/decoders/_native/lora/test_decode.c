/*
 * test_decode.c — minimal harness for the LoRa decoder.
 *
 * Usage: ./test_decode <file.cu8> <sample_rate> <bandwidth> <sf>
 *
 * Example:
 *   ./test_decode meshtastic_30s_913_5mhz_2_4msps.cu8 2400000 250000 11
 *
 * Prints summary stats and any decoded packets as hex + ASCII.
 *
 * Copyright (c) 2026, Off by One. BSD-3-Clause.
 */

#include "lora_demod.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

static int n_decoded = 0;

static void on_packet(const lora_decoded_t *pkt, void *userdata) {
    (void)userdata;
    n_decoded++;
    printf("\n[pkt %d] len=%u cr=%u crc=%s rssi=%.1f snr=%.1f cfo=%.0fHz "
           "@sample %" PRIu64 "\n",
           n_decoded, pkt->payload_len, pkt->cr,
           pkt->crc_ok ? "ok" : "FAIL",
           pkt->rssi_db, pkt->snr_db, pkt->cfo_hz,
           pkt->sample_offset);
    printf("  hex: ");
    for (int i = 0; i < pkt->payload_len && i < 64; i++) {
        printf("%02x", pkt->payload[i]);
    }
    if (pkt->payload_len > 64) printf("...");
    printf("\n  asc: ");
    for (int i = 0; i < pkt->payload_len && i < 64; i++) {
        char c = pkt->payload[i];
        putchar((c >= 0x20 && c < 0x7F) ? c : '.');
    }
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <file.cu8> <sample_rate> <bandwidth> <sf> [mix_freq_hz]\n"
            "  bandwidth: 125000 | 250000 | 500000\n"
            "  sf: 7..12\n"
            "  mix_freq_hz: optional digital downconversion frequency.\n"
            "      Set to (capture_freq - signal_freq). E.g. capture at\n"
            "      913.5 MHz, LoRa at 913.125 MHz: mix_freq_hz = 375000\n",
            argv[0]);
        return 1;
    }
    const char *path = argv[1];
    uint32_t sample_rate = (uint32_t)atoi(argv[2]);
    uint32_t bandwidth = (uint32_t)atoi(argv[3]);
    uint8_t sf = (uint8_t)atoi(argv[4]);
    int32_t mix_freq = (argc >= 6) ? atoi(argv[5]) : 0;

    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("fopen"); return 1; }

    /* Default Meshtastic sync word is 0x2B */
    lora_config_t cfg = {
        .sample_rate_hz = sample_rate,
        .bandwidth = (lora_bw_t)bandwidth,
        .sf = sf,
        .sync_word = 0x2B,
        .has_crc_default = 1,
        .ldro = 0,
        .mix_freq_hz = mix_freq,
    };
    fprintf(stderr,
        "Config: sample_rate=%u Hz  bw=%u Hz  sf=%u  sync=0x2B  mix=%+d Hz\n",
        sample_rate, bandwidth, sf, mix_freq);
    lora_demod_t *d = lora_demod_new(&cfg, on_packet, NULL);
    if (!d) { fprintf(stderr, "lora_demod_new failed\n"); return 1; }

    /* Stream the file in 256KB chunks of cu8. */
    enum { CHUNK_BYTES = 256 * 1024 };
    static uint8_t buf[CHUNK_BYTES];
    uint64_t total_bytes = 0;
    for (;;) {
        size_t n = fread(buf, 1, sizeof(buf), fp);
        if (n == 0) break;
        /* n is bytes; complex samples = n/2 (I,Q pair = 2 bytes) */
        lora_demod_process_cu8(d, buf, n / 2);
        total_bytes += n;
    }
    fclose(fp);

    lora_demod_stats_t stats;
    lora_demod_get_stats(d, &stats);
    printf("\n--- summary ---\n");
    printf("  bytes read         : %" PRIu64 "\n", total_bytes);
    printf("  samples processed  : %" PRIu64 "\n", stats.samples_processed);
    printf("  detect attempts    : %" PRIu64 "\n", stats.detect_attempts);
    printf("  detect above gate  : %" PRIu64 " (%.2f%%)\n",
           stats.detect_above_gate,
           stats.detect_attempts ?
             100.0 * stats.detect_above_gate / stats.detect_attempts : 0);
    printf("  detect peak max    : %.1f\n", stats.detect_peak_mag_max);
    printf("  detect max run     : %u  (need %u for preamble lock)\n",
           stats.detect_max_run, 8);
    printf("  preambles found    : %u\n", stats.preambles_found);
    printf("  syncwords matched  : %u\n", stats.syncwords_matched);
    printf("  headers decoded    : %u\n", stats.headers_decoded);
    printf("  headers failed     : %u\n", stats.headers_failed);
    printf("  packets decoded ok : %u\n", stats.packets_decoded);
    printf("  packets crc fail   : %u\n", stats.packets_crc_failed);

    lora_demod_free(d);
    return 0;
}
