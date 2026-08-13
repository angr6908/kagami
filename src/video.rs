//! Video playback delegated to libvlc, rendered natively. libvlc draws straight
//! into an `NSView` we add over the eframe/wgpu window (`set_nsobject`), so the
//! video lives on a CALayer that macOS animates smoothly — native fullscreen
//! behaves like IINA, and snapshots come out at full source resolution. The view
//! is mouse-transparent (`hitTest:` -> nil), so egui keeps owning all input.

use anyhow::{Result, anyhow};
use eframe::egui;
use objc2::rc::Retained;
use objc2::{MainThreadMarker, MainThreadOnly, class, define_class, msg_send};
use objc2_app_kit::{NSApplication, NSAutoresizingMaskOptions, NSView, NSWindowOrderingMode};
use objc2_foundation::{NSPoint, NSRect, NSSize};
use std::cell::Cell;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mov", "m4v", "mkv", "webm", "avi", "wmv", "flv", "mpg", "mpeg", "ts", "m2ts", "3gp",
];
pub const SEEK_STEP: f64 = 5.0;
/// Minimum gap between seek commands handed to libvlc. Native slider events can
/// arrive much faster than the decoder can restart, so keep only the latest
/// target and send at a modest interactive cadence.
const SEEK_FLUSH_MS: u64 = 50;
/// While a seek is pending, poll often enough that throttling adds at most one
/// display frame of latency once the flush interval expires.
const SEEK_RETRY_MS: u64 = 16;
/// A sent seek remains authoritative until libvlc's playback clock actually
/// reaches the requested neighborhood. The minimum hold prevents the stale
/// pre-seek clock from being mistaken for an acknowledgement on small seeks;
/// the maximum prevents a broken media source from pinning the UI forever.
const SEEK_ACK_MIN_MS: u64 = 100;
const SEEK_ACK_MAX_MS: u64 = 5_000;
const SEEK_ACK_TOLERANCE_SECS: f64 = 0.35;
/// AppKit slider tracking may report the same resting target more than once
/// around mouse-up. Re-seeking within roughly one display frame is not
/// perceptible, but it does restart VLC's decoder and can replay the first GOP
/// after the seek. Treat those callbacks as one logical seek.
const SEEK_DUPLICATE_TOLERANCE_SECS: f64 = 0.02;
const SEEK_DUPLICATE_WINDOW_MS: u64 = 250;
const AUDIO_TRACK_RETRY_MS: u64 = 500;

#[derive(Clone, Copy)]
struct SentSeek {
    target: f64,
    origin: f64,
    at: Instant,
}

/// One selectable audio stream reported by libvlc.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioTrack {
    pub id: i32,
    pub name: String,
}

fn seek_clock_acknowledged(
    origin: f64,
    target: f64,
    actual: f64,
    elapsed_secs: f64,
    paused: bool,
) -> bool {
    let target_window_end =
        target + if paused { 0.0 } else { elapsed_secs } + SEEK_ACK_TOLERANCE_SECS;
    let near_requested_clock =
        actual >= target - SEEK_ACK_TOLERANCE_SECS && actual <= target_window_end;

    // For a non-trivial jump, require readback to cross the midpoint from the
    // pre-seek clock toward the target. This rejects a stale clock even when a
    // small tolerance window happens to overlap it.
    let delta = target - origin;
    let moved_toward_target = if delta.abs() <= SEEK_ACK_TOLERANCE_SECS {
        true
    } else {
        let midpoint = origin + delta * 0.5;
        if delta > 0.0 {
            actual >= midpoint
        } else {
            actual <= midpoint
        }
    };

    near_requested_clock && moved_toward_target
}

fn same_seek_target(a: f64, b: f64) -> bool {
    (a - b).abs() <= SEEK_DUPLICATE_TOLERANCE_SECS
}

fn effective_mute(user_muted: bool, scrubbing: bool, seek_suppressed: bool) -> bool {
    user_muted || scrubbing || seek_suppressed
}

pub fn is_video_file(path: &Path) -> bool {
    crate::archive::has_ext(path, VIDEO_EXTENSIONS)
}

/// The libvlc core, shareable across threads (libvlc is thread-safe).
struct Instance(*mut libvlc_instance_t);
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

/// Warm up the shared libvlc core on a background thread so the plugin scan
/// runs during app startup, not when the first video opens. The plugin path
/// is resolved here, on the main thread, before anything else reads the
/// environment.
pub fn preload() {
    ensure_plugin_path();
    std::thread::spawn(|| {
        let _ = VideoPlayer::shared_instance();
    });
}

define_class!(
    // A layer-backed, mouse-transparent host for libvlc's vout. Returning nil from
    // `hitTest:` lets clicks fall through to the egui layer (play/seek/fullscreen).
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "KagamiVideoView"]
    struct VideoView;

    impl VideoView {
        #[unsafe(method(hitTest:))]
        fn hit_test(&self, _point: NSPoint) -> *mut NSView {
            core::ptr::null_mut()
        }
    }
);

/// libvlc loads its codecs/demuxers/output as plugins at runtime. Point it at
/// the copy vendored next to the executable (see scripts/bundle-macos.sh) so the
/// shipped .app is self-contained; fall back to a system VLC for local dev.
fn ensure_plugin_path() {
    if std::env::var_os("VLC_PLUGIN_PATH").is_some() {
        return;
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("../Frameworks/plugins"));
        candidates.push(dir.join("Frameworks/plugins"));
        candidates.push(dir.join("plugins"));
    }
    // Dev runs (raw target/ binary) use the libs extracted into vendor/vlc.
    candidates.push(PathBuf::from(format!(
        "{}/vendor/vlc/plugins",
        env!("CARGO_MANIFEST_DIR")
    )));
    candidates.push(PathBuf::from("/opt/homebrew/lib/vlc/plugins"));
    candidates.push(PathBuf::from("/usr/local/lib/vlc/plugins"));
    if let Some(dir) = candidates.into_iter().find(|c| c.is_dir()) {
        // SAFETY: set once at startup, before any VLC/thread reads the environment.
        unsafe { std::env::set_var("VLC_PLUGIN_PATH", dir) };
    }
}

pub struct VideoPlayer {
    mp: *mut libvlc_media_player_t,
    // The native render surface; kept alive while playing and released, on the
    // main thread, once the background teardown has stopped the vout.
    view: ManuallyDrop<Retained<VideoView>>,
    /// Stream source serving libvlc's read callbacks (`open_bytes` /
    /// `open_range`); null for plain file media. Freed by the teardown worker,
    /// after the player has stopped.
    src: *mut StreamSrc,
    paused: bool,
    muted: bool,
    volume: f32,
    /// libvlc can briefly output pre-seek decoder data after a clock jump.
    /// Keep that implementation detail separate from the user's mute state so
    /// mute/volume changes made while seeking are restored correctly.
    seek_audio_suppressed: Cell<bool>,
    /// Audio stream descriptions are immutable for the life of this player.
    /// VLC may not expose them immediately after `play`, so retry an empty
    /// query at a low cadence and cache the first populated list.
    audio_tracks: Vec<AudioTrack>,
    next_audio_track_query: Instant,
    #[allow(dead_code)]
    scrubbing: bool,
    /// Seek target not yet handed to libvlc. Rapid seeks (key mashing, scrub
    /// drags) accumulate here and flush at most once per `SEEK_FLUSH_MS`, so
    /// libvlc never queues a backlog of demux/decode restarts.
    pending_seek: Option<f64>,
    /// Normalized position for a playback-bar seek. Keeping this alongside the
    /// time target lets slider seeks use VLC's direct `SET_POSITION` path
    /// instead of `SET_TIME`, whose VLC 3 implementation may fall back to a
    /// second position seek for demuxers that reject time seeking.
    pending_seek_fraction: Option<f32>,
    /// Last target sent to libvlc and when; the base for chained relative
    /// seeks and the throttle clock for the next flush.
    sent_seek: Cell<Option<SentSeek>>,
    /// Last seek handed to libvlc, retained independently of `sent_seek`.
    /// `sent_seek` is cleared as soon as VLC's clock acknowledges the jump,
    /// which can happen before AppKit delivers a duplicate end-of-tracking
    /// action. Keeping dispatch history separate prevents that late action
    /// from restarting the decoder at the same frame.
    last_dispatched_seek: Cell<Option<(f64, Instant)>>,
    /// Position to restore once libvlc's input is live — a `set_time` issued
    /// right after `play` lands before the input thread exists and is dropped.
    resume_at: Option<f64>,
    /// View rotation in clockwise quarter-turns (0..4): R turns right, E turns
    /// left, T resets.
    rotation: u8,
    /// Last applied (rotation, width, height); skips redundant relayout.
    last_layout: Option<(u8, u32, u32)>,
}

impl VideoPlayer {
    pub fn open(path: &Path) -> Result<Self> {
        let mtm =
            MainThreadMarker::new().ok_or_else(|| anyhow!("video must open on main thread"))?;
        let cpath = CString::new(path.to_string_lossy().into_owned())
            .map_err(|_| anyhow!("path contains a NUL byte"))?;
        let instance = Self::shared_instance()?;
        let media = unsafe { libvlc_media_new_path(instance, cpath.as_ptr()) };
        if media.is_null() {
            return Err(anyhow!("libvlc_media_new_path failed"));
        }
        Self::start(mtm, media, std::ptr::null_mut())
    }

    /// Play a video held entirely in memory (a decompressed archive entry).
    /// The buffer is served to libvlc through the `src_*` read/seek callbacks,
    /// so nothing is extracted to disk.
    pub fn open_bytes(data: Vec<u8>) -> Result<Self> {
        Self::open_stream(StreamSrc::Mem { data, pos: 0 })
    }

    /// Play `len` bytes of `path` starting at `start` — a stored
    /// (uncompressed) archive entry, pread straight out of the archive file.
    /// No decompression, no buffer, no temp file.
    pub fn open_range(path: &Path, start: u64, len: u64) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::open_stream(StreamSrc::File {
            file,
            start,
            len,
            pos: 0,
        })
    }

    fn open_stream(src: StreamSrc) -> Result<Self> {
        let mtm =
            MainThreadMarker::new().ok_or_else(|| anyhow!("video must open on main thread"))?;
        let instance = Self::shared_instance()?;
        let src = Box::into_raw(Box::new(src));
        let media = unsafe {
            libvlc_media_new_callbacks(
                instance,
                Some(src_open),
                Some(src_read),
                Some(src_seek),
                Some(src_close),
                src.cast(),
            )
        };
        if media.is_null() {
            unsafe { drop(Box::from_raw(src)) };
            return Err(anyhow!("libvlc_media_new_callbacks failed"));
        }
        Self::start(mtm, media, src)
    }

    /// One libvlc core shared by every player. `libvlc_new` loads the whole
    /// plugin bank — a multi-second scan — so it happens exactly once for the
    /// process life, kicked off early by `preload` so the cost overlaps app
    /// startup instead of stalling the first video open.
    fn shared_instance() -> Result<*mut libvlc_instance_t> {
        static INSTANCE: std::sync::OnceLock<Instance> = std::sync::OnceLock::new();
        let instance = INSTANCE
            .get_or_init(|| {
                ensure_plugin_path();
                let args = [
                    CString::new("--no-video-title-show").unwrap(),
                    CString::new("--quiet").unwrap(),
                ];
                let argv: Vec<*const c_char> = args.iter().map(|a| a.as_ptr()).collect();
                Instance(unsafe { libvlc_new(argv.len() as c_int, argv.as_ptr()) })
            })
            .0;
        if instance.is_null() {
            return Err(anyhow!("libvlc_new failed (is VLC installed?)"));
        }
        Ok(instance)
    }

    /// Shared tail of the `open*` constructors: loop option, player, native
    /// render view, play. Releases everything (including `src`) if a step fails.
    fn start(
        mtm: MainThreadMarker,
        media: *mut libvlc_media_t,
        src: *mut StreamSrc,
    ) -> Result<Self> {
        let release_all = |mp: *mut libvlc_media_player_t| unsafe {
            if !mp.is_null() {
                libvlc_media_player_release(mp);
            }
            if !src.is_null() {
                drop(Box::from_raw(src));
            }
        };

        let loop_opt = CString::new(":input-repeat=65535").unwrap();
        unsafe { libvlc_media_add_option(media, loop_opt.as_ptr()) };
        // Keep direct timeline seeks precise. VLC's `input-fast-seek` mode
        // deliberately favors speed over accuracy and can land on an earlier
        // keyframe before playback catches up to the requested timestamp. That
        // is useful for coarse thumbnail/scrub workloads, but for a single
        // playback-bar click it shows up as a short repeated/jumped segment.
        // We already coalesce/throttle slider traffic below, so leave VLC in
        // its precise seek mode here.

        let mp = unsafe { libvlc_media_player_new_from_media(media) };
        unsafe { libvlc_media_release(media) };
        if mp.is_null() {
            release_all(std::ptr::null_mut());
            return Err(anyhow!("libvlc_media_player_new_from_media failed"));
        }

        // Build the native render view and slot it behind the AppKit control bar
        // but above the wgpu layer, sized to (and auto-resizing with) the window.
        let view = match Self::make_view(mtm) {
            Some(v) => v,
            None => {
                release_all(mp);
                return Err(anyhow!("no window to attach the video view to"));
            }
        };
        unsafe { libvlc_media_player_set_nsobject(mp, Retained::as_ptr(&view) as *mut c_void) };

        if unsafe { libvlc_media_player_play(mp) } != 0 {
            view.removeFromSuperview();
            release_all(mp);
            return Err(anyhow!("libvlc_media_player_play failed"));
        }

        Ok(Self {
            mp,
            view: ManuallyDrop::new(view),
            src,
            paused: false,
            muted: false,
            volume: 1.0,
            seek_audio_suppressed: Cell::new(false),
            audio_tracks: Vec::new(),
            next_audio_track_query: Instant::now(),
            scrubbing: false,
            pending_seek: None,
            pending_seek_fraction: None,
            sent_seek: Cell::new(None),
            last_dispatched_seek: Cell::new(None),
            resume_at: None,
            rotation: 0,
            last_layout: None,
        })
    }

    fn make_view(mtm: MainThreadMarker) -> Option<Retained<VideoView>> {
        let app = NSApplication::sharedApplication(mtm);
        let window = app.keyWindow().or_else(|| app.mainWindow())?;
        let content = window.contentView()?;
        let bounds = content.bounds();
        let view: Retained<VideoView> = {
            let this = VideoView::alloc(mtm);
            unsafe { msg_send![this, initWithFrame: bounds] }
        };
        view.setWantsLayer(true);
        view.setAutoresizingMask(
            NSAutoresizingMaskOptions::ViewWidthSizable
                | NSAutoresizingMaskOptions::ViewHeightSizable,
        );
        // Positioned `Below` (relative to nil) puts it at the back of the subview
        // list, so the control bar added later stays on top of the video.
        content.addSubview_positioned_relativeTo(&view, NSWindowOrderingMode::Below, None);
        Some(view)
    }

    /// Keep the UI loop ticking while playing so the seek bar / clock advance —
    /// the video itself is driven by libvlc's own vout, not by egui repaints.
    pub fn update(&mut self, egui_ctx: &egui::Context) {
        if !self.paused {
            egui_ctx.request_repaint_after(Duration::from_millis(100));
        }
        if let Some(t) = self.resume_at {
            if unsafe { libvlc_media_player_get_time(self.mp) } >= 0 {
                self.resume_at = None;
                self.seek_to(t);
            }
            egui_ctx.request_repaint_after(Duration::from_millis(50));
        }
        if self.pending_seek.is_some() {
            self.flush_seek(false);
            egui_ctx.request_repaint_after(Duration::from_millis(SEEK_RETRY_MS));
        }
        // Poll even while paused: a paused seek otherwise gets only the frame
        // triggered by the slider callback, leaving its temporary audio mute
        // active until some unrelated UI event arrives.
        if self.sent_seek.get().is_some() {
            self.in_flight_seek_target();
            if self.sent_seek.get().is_some() {
                egui_ctx.request_repaint_after(Duration::from_millis(SEEK_RETRY_MS));
            }
        }
        self.layout();
    }

    /// Turn the view by `quarters` clockwise quarter-turns (3 = one turn
    /// counterclockwise), cycling through 0/90/180/270.
    pub fn rotate(&mut self, quarters: u8) {
        self.rotation = (self.rotation + quarters) % 4;
    }

    pub fn reset_rotation(&mut self) {
        self.rotation = 0;
    }

    /// Size and orient the native video view for the current rotation. For a
    /// quarter turn we hand libvlc a viewport with width/height swapped and spin
    /// the view about its centre, so libvlc re-fits the frame and the rotated
    /// result lands back inside the window — windowed or fullscreen alike, since
    /// the superview bounds are re-read every frame.
    fn layout(&mut self) {
        let (w, h) = {
            let Some(sv) = (unsafe { self.view.superview() }) else {
                return;
            };
            let b = sv.bounds();
            (b.size.width, b.size.height)
        };
        let key = (self.rotation, w.round() as u32, h.round() as u32);
        if self.last_layout == Some(key) {
            return;
        }
        self.last_layout = Some(key);

        let frame = if self.rotation % 2 == 1 {
            NSRect {
                origin: NSPoint::new((w - h) / 2.0, (h - w) / 2.0),
                size: NSSize::new(h, w),
            }
        } else {
            NSRect {
                origin: NSPoint::new(0.0, 0.0),
                size: NSSize::new(w, h),
            }
        };
        let angle = -(self.rotation as f64) * 90.0;
        let view: &VideoView = &self.view;
        // Apply frame + rotation together with implicit animations off, so live
        // resizes and fullscreen transitions track the window without a lag frame.
        unsafe {
            let _: () = msg_send![class!(CATransaction), begin];
            let _: () = msg_send![class!(CATransaction), setDisableActions: true];
            view.setFrameCenterRotation(0.0);
            view.setFrame(frame);
            view.setFrameCenterRotation(angle);
            let _: () = msg_send![class!(CATransaction), commit];
        }
    }

    /// Write the current frame to `path` (PNG by extension) at full source
    /// resolution — exactly what's on screen, decoded by libvlc's vout.
    pub fn save_snapshot(&self, path: &Path) -> Result<()> {
        let cpath = CString::new(path.to_string_lossy().into_owned())
            .map_err(|_| anyhow!("path contains a NUL byte"))?;
        let r = unsafe { libvlc_video_take_snapshot(self.mp, 0, cpath.as_ptr(), 0, 0) };
        if r == 0 {
            Ok(())
        } else {
            Err(anyhow!("libvlc_video_take_snapshot failed ({r})"))
        }
    }

    pub fn toggle_pause(&mut self) {
        self.set_paused(!self.paused);
    }

    fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
        unsafe { libvlc_media_player_set_pause(self.mp, paused as c_int) };
    }

    pub fn toggle_mute(&mut self) {
        self.muted = !self.muted;
        self.apply_effective_mute();
    }

    /// Set volume to `v`, clamped to [0, 1]. A positive volume unmutes,
    /// matching what most players do.
    pub fn set_volume(&mut self, v: f32) {
        self.volume = v.clamp(0.0, 1.0);
        if self.volume > 0.0 {
            self.muted = false;
        }
        unsafe { libvlc_audio_set_volume(self.mp, (self.volume * 100.0) as c_int) };
        self.apply_effective_mute();
    }

    pub fn volume(&self) -> f32 {
        self.volume
    }

    fn apply_effective_mute(&self) {
        let muted = effective_mute(self.muted, self.scrubbing, self.seek_audio_suppressed.get());
        unsafe { libvlc_audio_set_mute(self.mp, muted as c_int) };
    }

    fn suppress_seek_audio(&self) {
        if !self.seek_audio_suppressed.replace(true) {
            self.apply_effective_mute();
        }
    }

    fn restore_seek_audio(&self) {
        if self.seek_audio_suppressed.replace(false) {
            self.apply_effective_mute();
        }
    }

    /// Audio streams currently known to VLC. Cache the first populated list;
    /// one player owns one media input, so its stream descriptions do not
    /// change during playback.
    pub fn audio_tracks(&mut self) -> &[AudioTrack] {
        if !self.audio_tracks.is_empty() || Instant::now() < self.next_audio_track_query {
            return &self.audio_tracks;
        }
        self.next_audio_track_query = Instant::now() + Duration::from_millis(AUDIO_TRACK_RETRY_MS);

        let head = unsafe { libvlc_audio_get_track_description(self.mp) };
        let mut tracks = Vec::new();
        let mut node = head;
        while !node.is_null() {
            let track = unsafe { &*node };
            let name = if track.psz_name.is_null() {
                format!("Audio Track {}", track.i_id)
            } else {
                unsafe { CStr::from_ptr(track.psz_name) }
                    .to_string_lossy()
                    .into_owned()
            };
            tracks.push(AudioTrack {
                id: track.i_id,
                name,
            });
            node = track.p_next;
        }
        if !head.is_null() {
            unsafe { libvlc_track_description_list_release(head) };
        }
        if !tracks.is_empty() {
            self.audio_tracks = tracks;
        }
        &self.audio_tracks
    }

    pub fn current_audio_track(&self) -> i32 {
        unsafe { libvlc_audio_get_track(self.mp) }
    }

    pub fn set_audio_track(&mut self, id: i32) {
        if self.audio_tracks.iter().any(|track| track.id == id) {
            unsafe {
                libvlc_audio_set_track(self.mp, id);
            }
        }
    }

    /// Seek relative to the newest requested target (not the possibly stale
    /// playback clock), so mashed presses chain into one accumulated jump.
    pub fn seek_by(&mut self, delta: f64) {
        let base = self.pending_seek.or_else(|| self.in_flight_seek_target());
        let t = (base.unwrap_or_else(|| self.vlc_position()) + delta).max(0.0);
        self.seek_to(t);
    }

    pub fn seek_to(&mut self, target: f64) {
        // An explicit user seek supersedes a deferred resume position. Without
        // this, a click made while a newly opened input is becoming ready can
        // be followed by the old resume target on the next update.
        self.resume_at = None;
        let duration = self.duration();
        let target = if duration > 0.0 {
            target.clamp(0.0, duration)
        } else {
            target.max(0.0)
        };
        self.pending_seek = Some(target);
        self.pending_seek_fraction = None;
        self.flush_seek(false);
    }

    /// Seek from a timeline fraction supplied by the playback bar. This keeps
    /// the request in the same coordinate system all the way into VLC instead
    /// of converting fraction -> duration -> time and asking the demuxer to
    /// translate it back again. In VLC 3, `SET_TIME` may itself fall back to a
    /// `SET_POSITION` request, so using position directly also guarantees one
    /// demux seek for one playback-bar action.
    pub fn seek_to_fraction(&mut self, fraction: f64) {
        self.resume_at = None;
        let fraction = fraction.clamp(0.0, 1.0);
        let duration = self.duration();
        let target = if duration > 0.0 {
            fraction * duration
        } else {
            0.0
        };
        self.pending_seek = Some(target);
        self.pending_seek_fraction = Some(fraction as f32);
        self.flush_seek(false);
    }

    /// Hand the pending target to libvlc, at most once per `SEEK_FLUSH_MS`
    /// unless forced. Last value wins; superseded targets are never sent.
    fn flush_seek(&mut self, force: bool) {
        let Some(target) = self.pending_seek else {
            return;
        };
        let sent = self.sent_seek.get();

        // AppKit can report the resting slider value again at the end of its
        // tracking loop. Do not key this guard off `sent_seek`: playback-clock
        // readback clears that as soon as VLC acknowledges the first jump,
        // sometimes before the duplicate UI action arrives. A second decoder
        // restart at the same timestamp is what makes the first scene after a
        // seek visibly play twice.
        if self.last_dispatched_seek.get().is_some_and(|(last, at)| {
            at.elapsed() < Duration::from_millis(SEEK_DUPLICATE_WINDOW_MS)
                && same_seek_target(last, target)
        }) {
            self.pending_seek = None;
            self.pending_seek_fraction = None;
            return;
        }

        let throttled =
            sent.is_some_and(|sent| sent.at.elapsed() < Duration::from_millis(SEEK_FLUSH_MS));
        if throttled && !force {
            return;
        }
        let fraction = self.pending_seek_fraction;
        self.pending_seek = None;
        self.pending_seek_fraction = None;
        let now = Instant::now();
        self.sent_seek.set(Some(SentSeek {
            target,
            origin: self.vlc_position(),
            at: now,
        }));
        self.last_dispatched_seek.set(Some((target, now)));
        // Silence libvlc before restarting the demuxer/decoder. Without this,
        // a short packet from the old playback position can escape before the
        // audio clock catches up, heard as a click or repeated syllable.
        self.suppress_seek_audio();
        unsafe {
            if let Some(fraction) = fraction {
                libvlc_media_player_set_position(self.mp, fraction);
            } else {
                libvlc_media_player_set_time(self.mp, (target * 1000.0) as i64);
            }
        }
    }

    /// Return the last sent target until libvlc's clock demonstrates that the
    /// seek landed. Once acknowledged, clear it permanently so a later loop or
    /// clock discontinuity cannot resurrect an old optimistic position.
    fn in_flight_seek_target(&self) -> Option<f64> {
        let sent = self.sent_seek.get()?;
        let elapsed = sent.at.elapsed();
        if elapsed >= Duration::from_millis(SEEK_ACK_MAX_MS) {
            self.sent_seek.set(None);
            if self.pending_seek.is_none() {
                self.restore_seek_audio();
            }
            return None;
        }

        if elapsed >= Duration::from_millis(SEEK_ACK_MIN_MS) {
            let actual = self.vlc_position();
            if seek_clock_acknowledged(
                sent.origin,
                sent.target,
                actual,
                elapsed.as_secs_f64(),
                self.paused,
            ) {
                self.sent_seek.set(None);
                // A newer throttled request may still be waiting. Keep output
                // muted across that handoff so no old-position packet escapes
                // during the one-frame gap before it is dispatched.
                if self.pending_seek.is_none() {
                    self.restore_seek_audio();
                }
                return None;
            }
        }

        Some(sent.target)
    }

    /// libvlc has no separate keyframe seek, so scrubbing reuses the plain seek.
    /// Each intermediate seek restarts the decoder and spits out a burst of audio,
    /// so mute the output for the duration of the drag and restore it in `end_scrub`.
    #[allow(dead_code)] // the native slider seeks directly; kept for scrub UX
    pub fn scrub_to(&mut self, target: f64) {
        if !self.scrubbing {
            self.scrubbing = true;
            self.apply_effective_mute();
        }
        self.seek_to(target);
    }

    /// Finish a scrub: land on the final drag position, then undo the
    /// scrub-time mute, leaving the user's mute intact.
    #[allow(dead_code)]
    pub fn end_scrub(&mut self) {
        if self.scrubbing {
            self.scrubbing = false;
            self.flush_seek(true);
            self.apply_effective_mute();
        }
    }

    /// The position the player is at or headed to: an unflushed or in-flight
    /// seek target reads back immediately, so the seek bar tracks rapid
    /// presses without waiting on libvlc.
    pub fn position(&self) -> f64 {
        self.resume_at
            .or(self.pending_seek)
            .or_else(|| self.in_flight_seek_target())
            .unwrap_or_else(|| self.vlc_position())
    }

    /// Continue from `t` as soon as playback has started.
    pub fn resume_from(&mut self, t: f64) {
        self.resume_at = Some(t.max(0.0));
    }

    fn vlc_position(&self) -> f64 {
        let ms = unsafe { libvlc_media_player_get_time(self.mp) };
        if ms < 0 { 0.0 } else { ms as f64 / 1000.0 }
    }
    pub fn duration(&self) -> f64 {
        let ms = unsafe { libvlc_media_player_get_length(self.mp) };
        if ms < 0 { 0.0 } else { ms as f64 / 1000.0 }
    }
    pub fn is_paused(&self) -> bool {
        self.paused
    }
    pub fn is_muted(&self) -> bool {
        self.muted
    }
}

#[cfg(test)]
mod seek_tests {
    use super::*;

    #[test]
    fn stale_clock_does_not_ack_forward_seek() {
        assert!(!seek_clock_acknowledged(10.0, 50.0, 10.2, 0.2, false));
    }

    #[test]
    fn stale_clock_does_not_ack_backward_seek() {
        assert!(!seek_clock_acknowledged(50.0, 10.0, 50.2, 0.2, false));
    }

    #[test]
    fn landed_seek_acknowledges_while_playing() {
        assert!(seek_clock_acknowledged(10.0, 50.0, 50.2, 0.2, false));
    }

    #[test]
    fn paused_seek_requires_target_neighborhood() {
        assert!(!seek_clock_acknowledged(10.0, 50.0, 49.0, 2.0, true));
        assert!(seek_clock_acknowledged(10.0, 50.0, 50.1, 2.0, true));
    }

    #[test]
    fn duplicate_slider_targets_within_one_frame_are_equivalent() {
        assert!(same_seek_target(50.0, 50.015));
        assert!(!same_seek_target(50.0, 50.025));
    }

    #[test]
    fn internal_audio_suppression_never_overrides_user_mute() {
        assert!(effective_mute(true, false, false));
        assert!(effective_mute(false, true, false));
        assert!(effective_mute(false, false, true));
        assert!(!effective_mute(false, false, false));
    }
}

impl Drop for VideoPlayer {
    fn drop(&mut self) {
        // Detach the view now (main thread) so a newly opened video layers over a
        // clean window; the blocking stop happens off-thread so switching videos
        // doesn't stall the UI. VLC keeps the view alive through the raw pointer
        // until it has stopped, then releases it back on the main thread.
        self.view.removeFromSuperview();
        let job = Teardown {
            mp: self.mp,
            src: self.src,
            view: Retained::into_raw(unsafe { ManuallyDrop::take(&mut self.view) }),
        };
        std::thread::spawn(move || job.run());
    }
}

/// The blocking half of tearing a player down. `libvlc_media_player_stop` joins
/// VLC's decoder/output threads, so it runs on a worker thread instead of the
/// main one. Only reached after the owning `VideoPlayer` is gone.
struct Teardown {
    mp: *mut libvlc_media_player_t,
    src: *mut StreamSrc,
    view: *mut VideoView,
}

// The player handle and stream source are handed over exclusively; the view is
// only carried through to its main-thread release below, never used off-main.
unsafe impl Send for Teardown {}

impl Teardown {
    fn run(self) {
        unsafe {
            libvlc_media_player_stop(self.mp);
            libvlc_media_player_release(self.mp);
            if !self.src.is_null() {
                drop(Box::from_raw(self.src));
            }
            // The vout has quit and no longer touches the view; release it back on
            // the main thread, where AppKit requires NSView deallocation to happen.
            dispatch_async_f(main_queue(), self.view.cast(), release_view);
        }
    }
}

unsafe extern "C" fn release_view(view: *mut c_void) {
    drop(unsafe { Retained::from_raw(view.cast::<VideoView>()) });
}

#[repr(C)]
struct dispatch_queue_s {
    _private: [u8; 0],
}

fn main_queue() -> *mut dispatch_queue_s {
    (&raw const _dispatch_main_q).cast_mut()
}

unsafe extern "C" {
    static _dispatch_main_q: dispatch_queue_s;
    fn dispatch_async_f(
        queue: *mut dispatch_queue_s,
        context: *mut c_void,
        work: unsafe extern "C" fn(*mut c_void),
    );
}

/// Backing store for archive playback: libvlc pulls the media through the
/// `src_*` callbacks below instead of opening a path. The state is only
/// touched from libvlc's input thread, never from Rust while the stream lives.
enum StreamSrc {
    /// A decompressed archive entry held in memory.
    Mem { data: Vec<u8>, pos: usize },
    /// A byte range of a file on disk (a stored zip entry), served with pread
    /// — no decompression and no buffering beyond libvlc's own.
    File {
        file: std::fs::File,
        start: u64,
        len: u64,
        pos: u64,
    },
}

impl StreamSrc {
    fn len(&self) -> u64 {
        match self {
            StreamSrc::Mem { data, .. } => data.len() as u64,
            StreamSrc::File { len, .. } => *len,
        }
    }

    fn set_pos(&mut self, offset: u64) {
        match self {
            StreamSrc::Mem { pos, .. } => *pos = offset as usize,
            StreamSrc::File { pos, .. } => *pos = offset,
        }
    }
}

unsafe extern "C" fn src_open(
    opaque: *mut c_void,
    datap: *mut *mut c_void,
    sizep: *mut u64,
) -> c_int {
    let s = unsafe { &mut *opaque.cast::<StreamSrc>() };
    // The `:input-repeat` loop reopens the stream: rewind, don't reallocate.
    s.set_pos(0);
    unsafe {
        *datap = opaque;
        *sizep = s.len();
    }
    0
}

unsafe extern "C" fn src_read(opaque: *mut c_void, buf: *mut u8, len: usize) -> isize {
    let s = unsafe { &mut *opaque.cast::<StreamSrc>() };
    match s {
        StreamSrc::Mem { data, pos } => {
            let n = len.min(data.len().saturating_sub(*pos));
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().add(*pos), buf, n) };
            *pos += n;
            n as isize
        }
        StreamSrc::File {
            file,
            start,
            len: total,
            pos,
        } => {
            use std::os::unix::fs::FileExt;
            let n = len.min(total.saturating_sub(*pos) as usize);
            let out = unsafe { std::slice::from_raw_parts_mut(buf, n) };
            match file.read_at(out, *start + *pos) {
                Ok(n) => {
                    *pos += n as u64;
                    n as isize
                }
                Err(_) => -1,
            }
        }
    }
}

unsafe extern "C" fn src_seek(opaque: *mut c_void, offset: u64) -> c_int {
    let s = unsafe { &mut *opaque.cast::<StreamSrc>() };
    if offset > s.len() {
        return -1;
    }
    s.set_pos(offset);
    0
}

/// The source outlives the stream (freed by the teardown worker, after the
/// player has fully stopped), so closing the stream is a no-op.
unsafe extern "C" fn src_close(_opaque: *mut c_void) {}

#[allow(non_camel_case_types)]
enum libvlc_instance_t {}
#[allow(non_camel_case_types)]
enum libvlc_media_t {}
#[allow(non_camel_case_types)]
enum libvlc_media_player_t {}

#[repr(C)]
struct libvlc_track_description_t {
    i_id: c_int,
    psz_name: *mut c_char,
    p_next: *mut libvlc_track_description_t,
}

type MediaOpenCb = unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut u64) -> c_int;
type MediaReadCb = unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> isize;
type MediaSeekCb = unsafe extern "C" fn(*mut c_void, u64) -> c_int;
type MediaCloseCb = unsafe extern "C" fn(*mut c_void);

unsafe extern "C" {
    fn libvlc_new(argc: c_int, argv: *const *const c_char) -> *mut libvlc_instance_t;
    fn libvlc_media_new_path(
        inst: *mut libvlc_instance_t,
        path: *const c_char,
    ) -> *mut libvlc_media_t;
    fn libvlc_media_new_callbacks(
        inst: *mut libvlc_instance_t,
        open_cb: Option<MediaOpenCb>,
        read_cb: Option<MediaReadCb>,
        seek_cb: Option<MediaSeekCb>,
        close_cb: Option<MediaCloseCb>,
        opaque: *mut c_void,
    ) -> *mut libvlc_media_t;
    fn libvlc_media_add_option(md: *mut libvlc_media_t, opt: *const c_char);
    fn libvlc_media_release(md: *mut libvlc_media_t);
    fn libvlc_media_player_new_from_media(md: *mut libvlc_media_t) -> *mut libvlc_media_player_t;
    fn libvlc_media_player_release(mp: *mut libvlc_media_player_t);
    fn libvlc_media_player_play(mp: *mut libvlc_media_player_t) -> c_int;
    fn libvlc_media_player_stop(mp: *mut libvlc_media_player_t);
    fn libvlc_media_player_set_pause(mp: *mut libvlc_media_player_t, do_pause: c_int);
    fn libvlc_media_player_set_nsobject(mp: *mut libvlc_media_player_t, drawable: *mut c_void);
    fn libvlc_media_player_get_time(mp: *mut libvlc_media_player_t) -> i64;
    fn libvlc_media_player_set_time(mp: *mut libvlc_media_player_t, t: i64);
    fn libvlc_media_player_set_position(mp: *mut libvlc_media_player_t, position: f32);
    fn libvlc_media_player_get_length(mp: *mut libvlc_media_player_t) -> i64;
    fn libvlc_video_take_snapshot(
        mp: *mut libvlc_media_player_t,
        num: c_uint,
        path: *const c_char,
        width: c_uint,
        height: c_uint,
    ) -> c_int;
    fn libvlc_audio_set_volume(mp: *mut libvlc_media_player_t, volume: c_int) -> c_int;
    fn libvlc_audio_set_mute(mp: *mut libvlc_media_player_t, status: c_int);
    fn libvlc_audio_get_track_description(
        mp: *mut libvlc_media_player_t,
    ) -> *mut libvlc_track_description_t;
    fn libvlc_track_description_list_release(tracks: *mut libvlc_track_description_t);
    fn libvlc_audio_get_track(mp: *mut libvlc_media_player_t) -> c_int;
    fn libvlc_audio_set_track(mp: *mut libvlc_media_player_t, track: c_int) -> c_int;
}
