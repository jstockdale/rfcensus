#!/usr/bin/env bash
# rfcensus decoder installer for Debian/Ubuntu.
#
# Installs the external decoder binaries rfcensus subprocesses. Safe to
# rerun — skips anything already present.
#
# Usage:
#   ./scripts/install_decoders.sh           # core decoders
#   ./scripts/install_decoders.sh --all     # core + opt-in decoders

set -euo pipefail

ALL=false
if [[ "${1:-}" == "--all" ]]; then
    ALL=true
fi

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok() { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
skip() { printf '  \033[2m·\033[0m %s\n' "$*"; }

need_sudo() {
    if ! command -v sudo >/dev/null 2>&1; then
        echo "sudo required but not installed"
        exit 1
    fi
}

# Detect distribution
if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    DISTRO="$ID"
else
    DISTRO="unknown"
fi

install_apt() {
    say "Installing apt packages"
    need_sudo
    PKGS=(
        rtl-sdr
        rtl-433
        multimon-ng
    )
    if $ALL; then
        PKGS+=(hackrf direwolf)
    fi
    sudo apt-get update
    sudo apt-get install -y "${PKGS[@]}"
}

install_rtlamr() {
    if command -v rtlamr >/dev/null 2>&1; then
        skip "rtlamr already installed at $(command -v rtlamr)"
        # If the user has upstream rtlamr, they're missing the
        # r900 fast-path patch. Mention it but don't force an
        # upgrade — we can't easily tell which version they have
        # (the fork keeps the upstream module path so rtlamr
        # -version looks identical).
        skip "  to upgrade to the fork with the r900 fast-path:"
        skip "    go install github.com/jstockdale/rtlamr@latest"
        return
    fi
    say "Installing rtlamr (Go)"
    if ! command -v go >/dev/null 2>&1; then
        echo "Go compiler not found. Install golang first:"
        echo "  sudo apt install golang"
        echo "Then rerun this script."
        return 1
    fi
    # We use jstockdale/rtlamr instead of upstream bemasher/rtlamr.
    # The fork carries a single, narrowly-scoped performance patch
    # to r900/r900.go that skips the 4-FSK matched filter and
    # quantize passes when no preamble matches exist in the current
    # block. Empirically ~35× faster for the common "no traffic"
    # case. Intended for upstream PR; until then, the fork is
    # what rfcensus tests against.
    # See: https://github.com/jstockdale/rtlamr
    go install github.com/jstockdale/rtlamr@latest
    GOBIN="$(go env GOBIN)"
    if [[ -z "$GOBIN" ]]; then
        GOBIN="$(go env GOPATH)/bin"
    fi
    if [[ ! -x "$GOBIN/rtlamr" ]]; then
        echo "rtlamr not found at $GOBIN/rtlamr after go install"
        return 1
    fi
    ok "rtlamr installed at $GOBIN/rtlamr"
    ok "make sure $GOBIN is on your PATH"
}

install_rtl_ais() {
    if command -v rtl_ais >/dev/null 2>&1; then
        skip "rtl_ais already installed"
        return
    fi
    say "Building rtl_ais from source"
    need_sudo
    sudo apt-get install -y build-essential cmake pkg-config libusb-1.0-0-dev librtlsdr-dev
    tmp=$(mktemp -d)
    pushd "$tmp" >/dev/null
    git clone https://github.com/dgiardini/rtl-ais.git
    cd rtl-ais
    make
    sudo make install
    popd >/dev/null
    rm -rf "$tmp"
    ok "rtl_ais installed"
}

main() {
    case "$DISTRO" in
        ubuntu|debian|raspbian|linuxmint)
            install_apt
            install_rtlamr || true
            install_rtl_ais || true
            ;;
        *)
            echo "Unsupported distro '$DISTRO'."
            echo "Install these binaries manually:"
            echo "  rtl-sdr (rtl_test, rtl_power, rtl_fm)"
            echo "  rtl_433"
            echo "  rtlamr"
            echo "  rtl_ais"
            echo "  multimon-ng"
            echo "  hackrf (optional, for HackRF support)"
            echo "  direwolf (optional, for APRS decoding)"
            exit 1
            ;;
    esac

    say "Done. Verify with:"
    echo "  rfcensus doctor"
}

main "$@"
