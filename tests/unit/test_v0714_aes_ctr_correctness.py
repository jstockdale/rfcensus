"""v0.7.14: AES-CTR counter increment correctness.

REGRESSION TEST FOR THE BUG FIXED IN v0.7.14:

The software AES-CTR fallback in ``meshtastic_crypto.h`` previously
incremented the 32-bit counter (bytes [12..15] of the IV) in
LITTLE-ENDIAN order — incrementing byte 12 first and carrying into
13, 14, 15. NIST SP 800-38A requires BIG-ENDIAN counter increment
(carrying from the rightmost byte toward the left), and that's what
mbedtls and OpenSSL implement.

The on-air Meshtastic firmware uses mbedtls. Block 0 of every
packet decrypted correctly because the counter starts at zero and
no increment happens before block 0 is generated. Block 1 onwards
got the WRONG keystream — our counter reached 1 by setting
counter[12]=1, while the firmware's counter reached 1 by setting
counter[15]=1. Different counter blocks → totally different AES
output → totally different keystream → garbage plaintext from
byte 16 onward.

Symptom in real captures: text messages would decode correctly for
the first ~12 bytes (4 bytes of protobuf framing + ~12 bytes of
text fitting in the first AES block), then turn into binary garbage.

GROUND TRUTH STRATEGY:
We use Python's `cryptography` library as the AES-CTR reference
because:
  • It's a thin wrapper around OpenSSL (NIST-validated)
  • It increments the counter big-endian per SP 800-38A
  • It's the same primitive Meshtastic firmware uses (via mbedtls,
    which matches OpenSSL's increment order)
The test encrypts a Meshtastic-shaped packet with the reference,
then decrypts through our pipeline. Bit-perfect round-trip proves
our counter increment matches the wire format.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest


LIB_PATH = Path(
    "/home/claude/rfcensus/rfcensus/decoders/_native/meshtastic/libmeshtastic.so"
)


# Default Meshtastic primary-channel PSK (base64 "AQ==") expanded to
# the 16-byte AES-128 key the firmware's CryptoEngine uses for the
# default channel. Matches MESH_DEFAULT_PSK in
# rfcensus/decoders/_native/meshtastic/src/meshtastic_channel.h:30.
DEFAULT_KEY = bytes([
    0xd4, 0xf1, 0xbb, 0x3a, 0x20, 0x29, 0x07, 0x59,
    0xf0, 0xbc, 0xff, 0xab, 0xcf, 0x4e, 0x69, 0x01,
])


def _channel_hash(name: bytes, key: bytes) -> int:
    """Compute the Meshtastic channel hash byte:
       hash = XOR(name) ^ XOR(key)
    Must match meshComputeChannelHash() in meshtastic_channel.h."""
    h = 0
    for b in name: h ^= b
    for b in key:  h ^= b
    return h & 0xff


def _build_meshtastic_packet(
    psk: bytes,
    from_node: int,
    packet_id: int,
    plaintext: bytes,
    channel_hash: int,
    to: int = 0xFFFFFFFF,
    hop_limit: int = 3,
) -> bytes:
    """Construct a raw Meshtastic LoRa frame (16-byte header +
    encrypted payload) using Python cryptography's AES-CTR as the
    reference. Result must be decodable by our library iff our
    AES-CTR matches the reference.

    Header layout per Meshtastic firmware (16 bytes):
      [0..3]   to        (uint32 LE)
      [4..7]   from      (uint32 LE)
      [8..11]  id        (uint32 LE)
      [12]     flags     (hop_limit + want_ack + via_mqtt)
      [13]     channel_hash
      [14]     next_hop
      [15]     relay_node
    """
    from cryptography.hazmat.primitives.ciphers import (
        Cipher, algorithms, modes,
    )
    nonce = (
        struct.pack("<Q", packet_id)
        + struct.pack("<I", from_node)
        + b"\x00\x00\x00\x00"
    )
    assert len(nonce) == 16
    cipher = Cipher(algorithms.AES(psk), modes.CTR(nonce))
    enc = cipher.encryptor()
    ciphertext = enc.update(plaintext) + enc.finalize()

    flags = hop_limit & 0x07
    header = (
        struct.pack("<I", to)
        + struct.pack("<I", from_node)
        + struct.pack("<I", packet_id)
        + bytes([flags, channel_hash, 0, 0])
    )
    return header + ciphertext


@pytest.mark.skipif(
    not LIB_PATH.exists(),
    reason="libmeshtastic.so not built",
)
class TestRoundTripDecryption:
    """Encrypt with Python `cryptography` (NIST/OpenSSL semantics)
    then decrypt through our pipeline. Round-trip must be
    bit-perfect for multi-block plaintexts. Pre-fix, blocks 1+
    would have come back as garbage — these tests would all fail
    except `test_single_block_decrypt`."""

    # Use a name whose channel hash is a nice round value to make
    # debugging easier; "LongFast" famously hashes to 0x08.
    CHANNEL_NAME = "LongFast"

    def _decode(self, raw: bytes):
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
        dec = MeshtasticDecoder("MEDIUM_FAST")
        dec.add_channel(name=self.CHANNEL_NAME, psk=DEFAULT_KEY)
        return dec.decode(raw)

    def _make_packet(self, from_node: int, packet_id: int,
                     plaintext: bytes) -> bytes:
        ch_hash = _channel_hash(self.CHANNEL_NAME.encode(), DEFAULT_KEY)
        return _build_meshtastic_packet(
            DEFAULT_KEY, from_node, packet_id, plaintext,
            channel_hash=ch_hash,
        )

    def test_single_block_decrypt(self) -> None:
        """One AES block of plaintext (16 bytes). This block is
        always correct, even with the v0.7.13 bug — useful as a
        baseline that the test infrastructure works."""
        plaintext = b"AAAAAAAAAAAAAAAA"   # 16 bytes = exactly one block
        raw = self._make_packet(0x12345678, 0xABCDEF01, plaintext)
        result = self._decode(raw)
        assert result.decrypted, "single-block decode failed"
        assert result.plaintext == plaintext

    def test_two_block_decrypt_matches_reference(self) -> None:
        """Two AES blocks of plaintext (32 bytes). With the v0.7.13
        bug, block 1 (bytes 16..31) would have been garbage. With
        the v0.7.14 fix this must match exactly."""
        plaintext = b"ABCDEFGHIJKLMNOP" * 2     # 32 bytes
        raw = self._make_packet(0x99BC7160, 0x10000001, plaintext)
        result = self._decode(raw)
        assert result.decrypted
        assert result.plaintext == plaintext, (
            f"two-block plaintext mismatch (counter-increment bug?)\n"
            f"  expected: {plaintext.hex()}\n"
            f"  got:      {result.plaintext.hex()}"
        )

    def test_long_text_message_decrypts_cleanly(self) -> None:
        """The exact failure mode the user observed: a long text
        message that previously turned to garbage after byte 16."""
        plaintext = (
            b"\x08\x01\x12\x70Just finished cooking 300 hotdogs and "
            b"probably 200 cheese burgers for the district 4 "
            b"challenger little league jamboree."
        )
        raw = self._make_packet(0xDB579BA4, 0x12345678, plaintext)
        result = self._decode(raw)
        assert result.decrypted
        assert result.plaintext == plaintext

    def test_eight_block_decrypt(self) -> None:
        """128 bytes = 8 blocks. Stresses the counter increment
        repeatedly (counter goes 0→1→2→3→4→5→6→7). The v0.7.13
        bug would have produced wrong output for blocks 1-7."""
        plaintext = bytes(range(128))   # 0x00..0x7f
        raw = self._make_packet(0x6985C590, 0xDEADBEEF, plaintext)
        result = self._decode(raw)
        assert result.decrypted
        assert result.plaintext == plaintext

    def test_all_block_boundaries_correct(self) -> None:
        """Test every length from 1 to 64 bytes. Catches edge cases
        at AES block boundaries (15, 16, 17, 31, 32, 33, ...)."""
        for length in range(1, 65):
            plaintext = bytes(range(length))
            raw = self._make_packet(0x244366DB, 0xCAFE0000 + length,
                                     plaintext)
            result = self._decode(raw)
            assert result.decrypted, f"len={length} not decrypted"
            assert result.plaintext == plaintext, (
                f"len={length} mismatch:\n"
                f"  expected: {plaintext.hex()}\n"
                f"  got:      {result.plaintext.hex()}"
            )


class TestCounterIncrementDirection:
    """Direct verification that the counter increments BIG-ENDIAN,
    using the reference cryptography library to confirm what the
    canonical behavior actually is. If someone reverses the loop
    in meshtastic_crypto.h again (perhaps thinking 'this looks
    wrong let me fix it'), the round-trip tests above will fail —
    this test ensures the test infrastructure itself is sound."""

    def test_reference_lib_uses_big_endian_counter(self) -> None:
        """Sanity check: confirm Python's cryptography library
        increments big-endian. Block 1's keystream must equal
        AES_ENC(IV with byte 15 = 1), NOT AES_ENC(IV with byte
        12 = 1). If the reference behavior ever changes, this
        test fails first and tells us why before the round-trip
        tests start failing mysteriously."""
        from cryptography.hazmat.primitives.ciphers import (
            Cipher, algorithms, modes,
        )

        # Encrypt 32 zero bytes with all-zero key & nonce
        key = b"\x00" * 16
        nonce = b"\x00" * 16
        ref_ks = (
            Cipher(algorithms.AES(key), modes.CTR(nonce))
            .encryptor().update(b"\x00" * 32)
        )
        # ref_ks[0:16]   = AES_ENC(all zeros)
        # ref_ks[16:32]  = AES_ENC(counter incremented once)

        # Compute big-endian +1 counter manually
        be_counter = b"\x00" * 15 + b"\x01"
        be_block = (
            Cipher(algorithms.AES(key), modes.ECB())
            .encryptor().update(be_counter)
        )

        # Compute the WRONG little-endian +1 counter (the v0.7.13 bug)
        le_counter = b"\x00" * 12 + b"\x01" + b"\x00" * 3
        le_block = (
            Cipher(algorithms.AES(key), modes.ECB())
            .encryptor().update(le_counter)
        )

        assert ref_ks[16:32] == be_block, (
            "reference cryptography lib is not big-endian — test "
            "infrastructure is broken; investigate before trusting "
            "any round-trip test results"
        )
        assert ref_ks[16:32] != le_block, (
            "big-endian and little-endian counters happen to "
            "produce same keystream — pick a different test key"
        )
