#!/bin/sh
# Build a standalone, audio-decoding FFmpeg without external codec libraries.
# Usage: sh scripts/build_ffmpeg.sh SOURCE_DIRECTORY OUTPUT_DIRECTORY [JOBS] [audio|shared]
# Requires a MinGW-w64 compiler and GNU make on PATH. Sources stay unmodified.
set -eu
source_dir=$(cd "$1" && pwd)
mkdir -p "$2"
output_dir=$(cd "$2" && pwd)
# Native MinGW make cannot resolve MSYS /c/... paths written by configure.
if command -v cygpath >/dev/null 2>&1; then
    source_dir=$(cygpath -m "$source_dir")
    output_dir=$(cygpath -m "$output_dir")
fi
jobs=${3:-4}
case "$jobs" in ''|*[!0-9]*|0) echo 'JOBS must be a positive integer' >&2; exit 2;; esac
mode=${4:-audio}
case "$mode" in
    audio) linkage='--disable-shared --enable-static'; programs='--disable-avdevice'; ldflags=-static; target=ffmpeg.exe;;
    shared) linkage='--enable-shared --disable-static'; programs='--disable-programs --enable-avdevice'; ldflags=-static-libgcc; target=all;;
    *) echo 'Mode must be audio or shared' >&2; exit 2;;
esac
cd "$output_dir"
# These variables intentionally contain configure switches, never paths.
"$source_dir/configure" \
    --target-os=mingw32 --arch=x86_64 \
    --disable-autodetect --disable-network --disable-x86asm \
    --disable-doc --disable-debug $linkage \
    --disable-ffplay --disable-ffprobe $programs \
    --disable-indevs --disable-outdevs --disable-hwaccels \
    --enable-w32threads --disable-pthreads \
    --disable-encoders --enable-encoder=pcm_s16le \
    --disable-muxers --enable-muxer=wav \
    --disable-filters --enable-filter=aresample,aformat,anull \
    --disable-decoders \
    --enable-decoder=aac,aac_fixed,aac_latm,ac3,ac3_fixed,eac3,alac,ape,atrac1,atrac3,atrac3p,cook,dca,flac,mp1,mp1float,mp2,mp2float,mp3,mp3float,mp3adu,mp3adufloat,opus,vorbis,tta,wavpack,wmalossless,wmapro,wmav1,wmav2,wmavoice,pcm_alaw,pcm_mulaw,pcm_s8,pcm_u8,pcm_s16be,pcm_s16le,pcm_s24be,pcm_s24le,pcm_s32be,pcm_s32le,pcm_f32be,pcm_f32le,pcm_f64be,pcm_f64le,adpcm_ima_wav,adpcm_ms,amrnb,amrwb,gsm,gsm_ms \
    --extra-ldflags="$ldflags"
# configure canonicalizes paths back to MSYS form. Adapt only generated files
# for native Windows make; the corresponding FFmpeg source stays unchanged.
if command -v cygpath >/dev/null 2>&1; then
    printf 'include %s/Makefile\n' "$source_dir" > Makefile
    awk -v src="$source_dir" '
        /^SRC_PATH=/ { $0 = "SRC_PATH=" src }
        /^SRC_LINK=/ { $0 = "SRC_LINK=" src }
        { print }
    ' ffbuild/config.mak > ffbuild/config.mak.tmp
    mv ffbuild/config.mak.tmp ffbuild/config.mak
fi
# Archive builds must not inherit an enclosing application's Git revision.
revision=$(cat "$source_dir/RELEASE")
revision="$revision" "$source_dir/ffbuild/version.sh" "$source_dir" libavutil/ffversion.h ""
make -j"$jobs" V=1 REVISION="$revision" "$target"
if [ "$mode" = audio ]; then
    ./ffmpeg.exe -version
    ./ffmpeg.exe -L
fi
