#!/usr/bin/env bash
set -euo pipefail

VLC="$(cd "${1:-vendor/vlc}" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat > "$work/gen.c" <<'EOF'
#include <dlfcn.h>
#include <stdio.h>

typedef void *(*vlc_new_fn)(int, const char *const *);
typedef void (*vlc_release_fn)(void *);

int main(int argc, char **argv) {
    if (argc < 2) {
        return 2;
    }
    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "%s\n", dlerror());
        return 1;
    }
    vlc_new_fn vlc_new = (vlc_new_fn)dlsym(lib, "libvlc_new");
    vlc_release_fn vlc_release = (vlc_release_fn)dlsym(lib, "libvlc_release");
    const char *args[] = {"--reset-plugins-cache", "--quiet"};
    void *instance = vlc_new(2, args);
    if (!instance) {
        fprintf(stderr, "libvlc_new failed\n");
        return 1;
    }
    vlc_release(instance);
    return 0;
}
EOF

cc -o "$work/gen" "$work/gen.c"
before="$(stat -f %m "$VLC/plugins/plugins.dat" 2>/dev/null || echo 0)"
VLC_PLUGIN_PATH="$VLC/plugins" DYLD_LIBRARY_PATH="$VLC/lib" "$work/gen" "$VLC/lib/libvlc.dylib"
after="$(stat -f %m "$VLC/plugins/plugins.dat" 2>/dev/null || echo 0)"
if [[ "$after" == "0" || "$after" == "$before" ]]; then
    echo "plugins.dat was not regenerated in $VLC/plugins" >&2
    exit 1
fi
codesign --force --sign - "$VLC/plugins/plugins.dat"
echo "regenerated $VLC/plugins/plugins.dat"
