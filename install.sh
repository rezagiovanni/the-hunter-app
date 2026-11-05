#!/bin/sh
# The Hunter — installer.
#
# Usage:  curl -fsSL <url>/install.sh | bash
#
# Contract (PRD FR-1, FR-2, FR-3):
#
#   * POSIX sh. No bashisms, because /bin/sh may be dash.
#   * NEVER uses sudo. If something needs root, it prints the exact command and
#     stops. An installer that quietly escalates on a stranger's machine is not
#     acceptable.
#   * Nothing runs. It installs code, a venv, the model and a crontab block in
#     which every job is COMMENTED OUT. No data is fetched, no trade is placed,
#     nothing is scheduled to start.
#   * Idempotent. Re-running upgrades in place and never duplicates.
#   * The API key is NOT collected here. That happens in `hunter`, the CLI, once
#     the user chooses to run it.
#
# Environment overrides (used by the test harness):
#   HUNTER_HOME       install root        (default: ~/.local/share/the-hunter)
#   HUNTER_SRC        tarball path or URL (default: the release URL)
#   HUNTER_BIN_DIR    symlink location    (default: ~/.local/bin)
#   HUNTER_SKIP_MODEL skip the model download
#   HUNTER_DRY_RUN    print actions, change nothing

set -eu

VERSION="0.1.0"
RELEASE_URL="${HUNTER_SRC:-https://github.com/rezagiovanni/the-hunter-app/archive/refs/tags/v${VERSION}.tar.gz}"
MODEL_URL="${HUNTER_MODEL_URL:-https://github.com/rezagiovanni/the-hunter-app/releases/download/v${VERSION}/model.joblib}"

HUNTER_HOME="${HUNTER_HOME:-$HOME/.local/share/the-hunter}"
BIN_DIR="${HUNTER_BIN_DIR:-$HOME/.local/bin}"
DRY_RUN="${HUNTER_DRY_RUN:-0}"

APP_DIR="$HUNTER_HOME/app"
DATA_DIR="$HUNTER_HOME/data"
MARKET_DIR="$HUNTER_HOME/market"
VENV_DIR="$HUNTER_HOME/venv"
LOG_DIR="$HUNTER_HOME/logs"

TMP=""
cleanup() {
    [ -n "$TMP" ] && [ -d "$TMP" ] && rm -rf "$TMP" || true
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_RESET="$(printf '\033[0m')"; C_GREEN="$(printf '\033[32m')"
    C_RED="$(printf '\033[31m')";   C_YELLOW="$(printf '\033[33m')"
    C_DIM="$(printf '\033[2m')";    C_BOLD="$(printf '\033[1m')"
else
    C_RESET=""; C_GREEN=""; C_RED=""; C_YELLOW=""; C_DIM=""; C_BOLD=""
fi

step()  { printf '  %s\n' "$1"; }
ok()    { printf '  %s%s%s\n' "$C_GREEN" "[ok]" "$C_RESET"; }
info()  { printf '        %s%s%s\n' "$C_DIM" "$1" "$C_RESET"; }
warn()  { printf '  %s%s%s %s\n' "$C_YELLOW" "[warn]" "$C_RESET" "$1"; }

die() {
    printf '\n  %s%s%s %s\n\n' "$C_RED" "[FAIL]" "$C_RESET" "$1"
    shift
    for line in "$@"; do printf '        %s\n' "$line"; done
    printf '\n'
    exit 1
}

heading() {
    printf '\n  %s%s%s\n' "$C_BOLD" "$1" "$C_RESET"
    printf '  %s\n' "------------------------------------------------------------"
}

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
# Each check prints the exact command the user must run themselves.
# ---------------------------------------------------------------------------

need_python() {
    command -v python3 >/dev/null 2>&1 || die \
        "python3 is not installed" \
        "Install it first, for example:" \
        "  sudo apt install python3"
    PY="$(command -v python3)"
}

# The venv check must actually CREATE one. `python3 -m venv --help` exits 0 even
# when ensurepip is absent, so a help-text probe reports success on a machine
# where every venv then fails. This was a real finding on the target STB.
need_venv_support() {
    probe="$TMP/venvprobe"
    if ! "$PY" -m venv "$probe" >"$TMP/venvprobe.log" 2>&1; then
        pyver="$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "")"
        msg="$(cat "$TMP/venvprobe.log" 2>/dev/null | head -3)"
        die \
            "python3-venv is not available" \
            "python3 exists but cannot create virtual environments." \
            "${msg}" \
            "" \
            "Run this yourself (the installer will not use sudo):" \
            "  sudo apt update && sudo apt install python${pyver:+$pyver}-venv" \
            "  # if that package does not exist, try: sudo apt install python3-venv"
    fi
    rm -rf "$probe"
}

need_curl() {
    command -v curl >/dev/null 2>&1 || die \
        "curl is not installed" \
        "Install it first:" \
        "  sudo apt install curl"
}

need_basic_tools() {
    # mktemp and tar are in the same package family as the shell itself, but a
    # stripped container can lack them. Checked explicitly so the failure is a
    # sentence rather than "mktemp: not found" from line 137.
    for tool in mktemp tar; do
        command -v "$tool" >/dev/null 2>&1 || die \
            "$tool is not installed" \
            "Install it first:" \
            "  sudo apt install coreutils tar"
    done
}

need_tar() {
    command -v tar >/dev/null 2>&1 || die \
        "tar is not installed" \
        "Install it first:" \
        "  sudo apt install tar"
}

need_crontab() {
    if ! command -v crontab >/dev/null 2>&1; then
        warn "crontab is not installed; scheduled jobs will not be registered"
        info "install it later with: sudo apt install cron"
        HAVE_CRONTAB=0
    else
        HAVE_CRONTAB=1
    fi
}

check_prereqs() {
    heading "Checking prerequisites"
    need_basic_tools
    TMP="$(mktemp -d)"
    step "python3"
    need_python; ok
    step "python3-venv"
    need_venv_support; ok
    step "curl"
    need_curl; ok
    step "mktemp, tar"
    need_basic_tools; ok
    step "crontab"
    need_crontab
    printf '  arch: %s\n' "$(uname -m 2>/dev/null || echo unknown)"
}

# ---------------------------------------------------------------------------
# Fetch and place the code
# ---------------------------------------------------------------------------
fetch_source() {
    heading "Installing the application"

    if [ "$DRY_RUN" = "1" ]; then
        step "would fetch from $RELEASE_URL"
        return 0
    fi

    mkdir -p "$APP_DIR" "$DATA_DIR" "$MARKET_DIR" "$LOG_DIR"

    src_tmp="$TMP/src"
    mkdir -p "$src_tmp"

    case "$RELEASE_URL" in
        http://*|https://*)
            step "downloading"
            if ! curl -fsSL "$RELEASE_URL" -o "$TMP/app.tar.gz"; then
                die "could not download $RELEASE_URL" \
                    "Check your network, or pass a local tarball:" \
                    "  HUNTER_SRC=/path/to/app.tar.gz sh install.sh"
            fi
            tar -xzf "$TMP/app.tar.gz" -C "$src_tmp" || die "the download is not a valid tarball"
            ;;
        *)
            [ -f "$RELEASE_URL" ] || die "no such file: $RELEASE_URL"
            step "using local tarball"
            tar -xzf "$RELEASE_URL" -C "$src_tmp" || die "not a valid tarball: $RELEASE_URL"
            ;;
    esac
    ok

    # A GitHub archive nests everything under one top-level directory.
    inner="$src_tmp"
    count=$(ls -1 "$src_tmp" | wc -l | tr -d ' ')
    if [ "$count" = "1" ] && [ -d "$src_tmp/$(ls -1 "$src_tmp" | head -1)" ]; then
        inner="$src_tmp/$(ls -1 "$src_tmp" | head -1)"
    fi

    # Replace the code, keep the data. This is the upgrade path too.
    step "placing code in $APP_DIR"
    if [ -d "$APP_DIR/scripts" ]; then
        # preserve nothing from the old code: the tarball is the source of truth
        rm -rf "$APP_DIR/scripts" "$APP_DIR/bin" "$APP_DIR/static"
        rm -f "$APP_DIR/requirements.lock" "$APP_DIR/config.yaml.example"
    fi
    for d in scripts bin static model; do
        [ -d "$inner/$d" ] && cp -R "$inner/$d" "$APP_DIR/" 2>/dev/null || true
    done
    for f in requirements.lock config.yaml.example README.md; do
        [ -f "$inner/$f" ] && cp "$inner/$f" "$APP_DIR/" || true
    done
    [ -f "$APP_DIR/bin/hunter" ] && chmod +x "$APP_DIR/bin/hunter" || true
    ok
}

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
create_venv() {
    heading "Setting up the Python environment"

    if [ "$DRY_RUN" = "1" ]; then
        step "would create a venv at $VENV_DIR"
        return 0
    fi

    if [ ! -x "$VENV_DIR/bin/python" ]; then
        step "creating the virtual environment"
        "$PY" -m venv "$VENV_DIR" || die "could not create a venv at $VENV_DIR"
        ok
    else
        step "virtual environment already exists"
        ok
    fi

    step "installing dependencies"
    if [ -f "$APP_DIR/requirements.lock" ]; then
        # Exact pins. Nothing resolves, so the stack is the one that was tested.
        "$VENV_DIR/bin/pip" install --quiet --upgrade pip >/dev/null 2>&1 || true
        if ! "$VENV_DIR/bin/pip" install --quiet -r "$APP_DIR/requirements.lock"; then
            die "dependency installation failed" \
                "Retry with output visible:" \
                "  $VENV_DIR/bin/pip install -r $APP_DIR/requirements.lock"
        fi
    else
        die "requirements.lock is missing from the release" \
            "The tarball is incomplete; re-download it."
    fi
    ok
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
fetch_model() {
    heading "Fetching the ML model"

    if [ "${HUNTER_SKIP_MODEL:-0}" = "1" ]; then
        step "skipped (HUNTER_SKIP_MODEL=1)"
        return 0
    fi

    if [ "$DRY_RUN" = "1" ]; then
        step "would download the model"
        return 0
    fi

    if [ -f "$APP_DIR/model/model.joblib" ]; then
        step "model already present"
        ok
        return 0
    fi

    step "downloading model.joblib"
    if curl -fsSL "$MODEL_URL" -o "$APP_DIR/model/model.joblib"; then
        ok
    else
        # Not fatal: the app runs, only the ML veto layer is inactive. It fails
        # OPEN by design (no predictions -> no vetoes), so this is a warning.
        warn "could not download the model"
        info "the app will run without the ML veto layer"
        info "retry later:  curl -fsSL $MODEL_URL -o $APP_DIR/model/model.joblib"
    fi
}

# ---------------------------------------------------------------------------
# Crontab — all jobs disabled
# ---------------------------------------------------------------------------
install_crontab() {
    heading "Registering scheduled jobs"

    if [ "$HAVE_CRONTAB" = "0" ]; then
        warn "skipped: crontab is not installed"
        return 0
    fi

    if [ "$DRY_RUN" = "1" ]; then
        step "would register 6 jobs, all disabled"
        return 0
    fi

    if "$VENV_DIR/bin/python" "$APP_DIR/scripts/crontab_manager.py" >/dev/null 2>&1; then
        :   # self-check passes; not the operation we want here
    fi

    # Install through the CLI's own manager, so there is one implementation of
    # the block format and one place where the rules live.
    if "$VENV_DIR/bin/python" "$APP_DIR/scripts/cli.py" cron install >/dev/null 2>&1; then
        step "registered 6 jobs"
        info "ALL DISABLED. Nothing will run until you enable them."
        ok
    else
        warn "could not register the crontab block"
        info "do it later with:  hunter cron install"
    fi
}

# ---------------------------------------------------------------------------
# The hunter command
# ---------------------------------------------------------------------------
link_command() {
    heading "Installing the command"

    if [ "$DRY_RUN" = "1" ]; then
        step "would link $BIN_DIR/hunter"
        return 0
    fi

    mkdir -p "$BIN_DIR"
    if ln -sf "$APP_DIR/bin/hunter" "$BIN_DIR/hunter"; then
        step "linked $BIN_DIR/hunter"
        ok
    else
        warn "could not link into $BIN_DIR"
        info "run the app directly:  $APP_DIR/bin/hunter"
        return 0
    fi

    ensure_path
}

# ---------------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------------
# Symlinking into ~/.local/bin is not enough on its own. Debian, Ubuntu and
# Armbian put this in the skeleton .profile:
#
#     if [ -d "$HOME/.local/bin" ] ; then
#         PATH="$HOME/.local/bin:$PATH"
#     fi
#
# That test runs when the profile is first read. If the directory did not exist
# yet - the usual case on a freshly flashed image - nothing is added, and every
# future shell has a PATH without ~/.local/bin. Creating the directory afterwards
# does not help, because the profile is not re-read. The result is a working
# install and a `hunter: command not found`.
#
# So this function fixes the PATH itself rather than telling the user to.
ensure_path() {
    case ":$PATH:" in
        *":$BIN_DIR:"*) return 0 ;;
    esac

    if [ "$DRY_RUN" = "1" ]; then
        step "would add $BIN_DIR to PATH in .profile"
        return 0
    fi

    PROFILE="$HOME/.profile"
    MARKER="# added by the-hunter installer"
    LINE="export PATH=\"$BIN_DIR:\$PATH\""

    if [ -f "$PROFILE" ] && grep -qF "$MARKER" "$PROFILE" 2>/dev/null; then
        step "$BIN_DIR already recorded in .profile"
    else
        {
            printf '\n%s\n%s\n' "$MARKER" "$LINE"
        } >> "$PROFILE" 2>/dev/null \
            && step "added $BIN_DIR to PATH in .profile" \
            || { warn "could not write to $PROFILE"; }
    fi

    # Make it work in the CURRENT shell too, so `hunter` runs immediately.
    # A script cannot change its parent's environment, so the best available
    # thing is to place the command somewhere that is already on PATH.
    if [ -w /usr/local/bin ] 2>/dev/null; then
        ln -sf "$APP_DIR/bin/hunter" /usr/local/bin/hunter 2>/dev/null \
            && step "also linked /usr/local/bin/hunter (already on PATH)"
    else
        printf '\n'
        warn "\"%shunter\" will not work in this shell yet"
        info "run this once, now:"
        info "  export PATH=\"$BIN_DIR:\$PATH\""
        info "then, and in every new shell, it will just work."
    fi
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
finish() {
    heading "Installed"
    printf '  %sNothing is running.%s\n\n' "$C_BOLD" "$C_RESET"
    printf '  Next:\n'
    printf '    1. %shunter%s          — enter your Indodax API key\n' "$C_BOLD" "$C_RESET"
    printf '    2. open the dashboard and use Settings to fetch data\n\n'
    printf '  %sapp    %s%s\n' "$C_DIM" "$APP_DIR" "$C_RESET"
    printf '  %sdata   %s%s\n' "$C_DIM" "$DATA_DIR" "$C_RESET"
    printf '  %sconfig %s%s\n' "$C_DIM" "$HUNTER_HOME/config.yaml" "$C_RESET"
    printf '  %slogs   %s%s\n' "$C_DIM" "$LOG_DIR" "$C_RESET"
    printf '\n'
    printf '  %sTo remove it later:  hunter uninstall%s\n\n' "$C_DIM" "$C_RESET"
}

main() {
    printf '\n  %sThe Hunter%s %s— installer\n' "$C_BOLD" "$C_RESET" "$C_DIM"
    [ "$DRY_RUN" = "1" ] && printf '  %s(DRY RUN — nothing will be changed)%s\n' "$C_YELLOW" "$C_RESET"
    printf '\n'

    check_prereqs
    fetch_source
    create_venv
    fetch_model
    install_crontab
    link_command
    finish
}

main "$@"
