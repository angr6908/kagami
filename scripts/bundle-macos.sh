#!/usr/bin/env bash
# Vendor libvlc and its plugins from a VLC.app into Kagami.app so the shipped
# app is self-contained and needs nothing installed on the user's machine.
#
# libvlc is not a single dylib: it dlopens ~hundreds of plugins (codecs,
# demuxers, the vmem output) at runtime. We copy VLC's `lib/` (libvlc +
# libvlccore + their deps) and `plugins/` verbatim, preserving their relative
# layout so the plugins' own @loader_path references to libvlccore still
# resolve; only the Kagami binary's reference to libvlc is repointed to @rpath.
# At runtime src/video.rs sets VLC_PLUGIN_PATH to the vendored plugins dir.
# Uses only macOS built-ins (otool/install_name_tool/codesign).
#
# Usage: bundle-macos.sh <mach-o-binary> <frameworks-dir> [vlc-macos-dir]
set -euo pipefail

# Source dir holding libvlc's lib/ and plugins/ (extracted from VLC.app).
BIN="$1"
FW="$2"
VLC="${3:-vendor/vlc}"

mkdir -p "$FW/lib" "$FW/plugins"
# Keep VLC's layout verbatim, symlinks included (libvlc.dylib -> libvlc.5.dylib),
# so we don't duplicate the dylibs; -type f below skips the symlinks when signing.
ditto "$VLC/lib" "$FW/lib"
ditto "$VLC/plugins" "$FW/plugins"
chmod -R u+w "$FW"

# Repoint the binary's libvlc reference to @rpath, and add rpaths so it finds
# Frameworks/lib whether the binary sits in a .app or in a loose folder.
libvlc_ref="$(otool -L "$BIN" | awk '/libvlc(\.[0-9]+)*\.dylib/{print $1; exit}')"
if [[ -n "${libvlc_ref:-}" ]]; then
    install_name_tool -change "$libvlc_ref" "@rpath/libvlc.dylib" "$BIN"
fi
install_name_tool -add_rpath "@executable_path/../Frameworks/lib" "$BIN" 2>/dev/null || true
install_name_tool -add_rpath "@executable_path/Frameworks/lib" "$BIN" 2>/dev/null || true

# Drop every build-time rpath (anything not @executable_path) so the shipped app
# resolves libvlc only from its own Frameworks/lib, never a build-host path.
otool -l "$BIN" | awk '/LC_RPATH/{f=1} f&&/ path /{print $2; f=0}' | while IFS= read -r rp; do
    case "$rp" in
        @executable_path*) ;;
        *) install_name_tool -delete_rpath "$rp" "$BIN" 2>/dev/null || true ;;
    esac
done

resigned_plugin=0
while IFS= read -r -d '' f; do
    if ! codesign -v "$f" 2>/dev/null; then
        codesign --force --sign - "$f"
        if [[ "$f" == "$FW/plugins/"*.dylib ]]; then
            resigned_plugin=1
        fi
    fi
done < <(find "$FW" -type f -print0)
if [[ "$resigned_plugin" == 1 ]]; then
    bash "$(dirname "$0")/vlc-cache-gen.sh" "$FW"
fi
codesign --force --sign - "$BIN"

echo "bundled $(find "$FW" -name '*.dylib' | wc -l | tr -d ' ') dylibs into $FW"
