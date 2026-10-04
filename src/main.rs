use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use std::cmp::Ordering as CmpOrd;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

mod archive;
use archive::{Item, READ_CHUNK, VideoSource, has_ext, is_archive_file, scan_archive};

mod pdf;
use pdf::{PdfImage, is_pdf_file, scan_pdf};

mod ass;

mod video;
use video::{SUBTITLE_EXTENSIONS, VideoPlayer, is_subtitle_file, is_video_file};

#[cfg(target_os = "macos")]
mod controls;

// The image folders live on OneDrive "Files On-Demand", so a file is usually a
// cloud placeholder until something reads it. Reading the whole file ("the
// touch") forces OneDrive to hydrate it to local disk. The dominant cost is that
// network download, so we pre-hydrate a wide look-ahead window in the background
// while giving the on-screen image first claim on bandwidth.
const HYDRATE_AHEAD: usize = 64;
const HYDRATE_BEHIND: usize = 64;
const HYDRATE_WORKERS: usize = 8;
// Decode + keep-as-texture only a small window around the current image; each
// decoded texture costs real RAM/VRAM.
const DECODE_AHEAD: usize = 3;
const DECODE_BEHIND: usize = 2;
const DECODE_WORKERS: usize = 3;
const PRIORITY_WORKERS: usize = 2;
const MAX_DECODE_RETRIES: usize = 3;
// How close (in browse-list slots) the current image must get to an un-expanded
// archive/PDF before it is read. Small, so opening a folder never downloads a
// container the viewer isn't actually approaching. A little look-ahead lets the
// pages be ready by the time navigation reaches them.
const EXPAND_AHEAD: usize = 3;
const EXPAND_BEHIND: usize = 1;
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"];
/// eframe Storage key for the saved playback-panel position.
const PANEL_POS_KEY: &str = "playback_panel_pos";

/// IINA persists the floating OSC using normalized coordinates: the horizontal
/// centre as a fraction of the video width and the bottom edge as a fraction of
/// the video height. Keeping the same representation makes resize/restoration
/// behavior match IINA instead of pinning the panel to old absolute pixels.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct PanelPosition {
    horizontal: f32,
    vertical: f32,
}

/// A fixed-size pool of worker threads pulling jobs off a shared channel. This
/// is all we needed from rayon: drop that dependency (and its crossbeam-deque /
/// rayon-core subtree) in favour of std threads over the crossbeam channel we
/// already use. Workers exit when the pool is dropped and the sender closes.
struct ThreadPool {
    tx: Option<Sender<Box<dyn FnOnce() + Send>>>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl ThreadPool {
    fn new(threads: usize, name: &'static str) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded::<Box<dyn FnOnce() + Send>>();
        let workers = (0..threads)
            .map(|i| {
                let rx = rx.clone();
                thread::Builder::new()
                    .name(format!("{name}-{i}"))
                    .spawn(move || {
                        while let Ok(job) = rx.recv() {
                            job();
                        }
                    })
                    .unwrap()
            })
            .collect();
        Self {
            tx: Some(tx),
            workers,
        }
    }

    fn spawn<F: FnOnce() + Send + 'static>(&self, job: F) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Box::new(job));
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.tx.take(); // close the channel so workers see the queue end
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Outcome of a decode task, so the coordinator can tell a real failure (count
/// it against the retry budget) from an abort because the user navigated away.
enum DecodeOutcome {
    Ok,
    Failed,
    Aborted,
}

struct SharedState {
    current_idx: AtomicUsize,
    image_count: AtomicUsize,
    /// Set when a newer pipeline supersedes this one, so its coordinator (which
    /// otherwise loops forever) exits instead of spinning on a stale list.
    cancelled: AtomicBool,
    /// Commands pushed by the native AppKit panel since the last frame, applied
    /// to the player during the next egui frame.
    #[cfg(target_os = "macos")]
    native_cmds: Mutex<Vec<NativeCmd>>,
    /// Pending native panel move using IINA's normalized OSC coordinates, set
    /// by the drag callback and applied to `playback_panel_pos` next frame.
    #[cfg(target_os = "macos")]
    native_panel_move: Mutex<Option<(f64, f64)>>,
}

/// A command pushed by the native AppKit panel (IINA's OSC) and applied to the
/// player on the next egui frame. The panel's target/action callbacks can't
/// reach `&mut KagamiApp`, so they enqueue these instead.
#[derive(Debug, Clone, Copy)]
enum NativeCmd {
    Seek(f64),
    Volume(f32),
    Play,
    SeekBack,
    SeekForward,
    Mute,
    Fullscreen,
    SaveFrame,
    AudioTrack(i32),
}

#[cfg(target_os = "macos")]
fn push_native_cmd(queue: &mut Vec<NativeCmd>, cmd: NativeCmd) {
    match (queue.last_mut(), cmd) {
        // Native sliders are continuous controls. If several absolute updates
        // arrive before egui gets a frame, only the newest value matters; keep
        // relative/toggle commands ordered around them.
        (Some(NativeCmd::Seek(current)), NativeCmd::Seek(next)) => *current = next,
        (Some(NativeCmd::Volume(current)), NativeCmd::Volume(next)) => *current = next,
        (_, cmd) => queue.push(cmd),
    }
}

#[cfg(target_os = "macos")]
fn native_controls_callbacks(
    shared: &Arc<SharedState>,
    ctx: &egui::Context,
) -> controls::ControlsCallbacks {
    let cmd = |cmd: NativeCmd| {
        let shared = Arc::clone(shared);
        let repaint = ctx.clone();
        Box::new(move || {
            push_native_cmd(&mut shared.native_cmds.lock().unwrap(), cmd);
            repaint.request_repaint();
        }) as Box<dyn Fn()>
    };

    controls::ControlsCallbacks {
        on_seek: {
            let shared = Arc::clone(shared);
            let repaint = ctx.clone();
            Box::new(move |fraction| {
                push_native_cmd(
                    &mut shared.native_cmds.lock().unwrap(),
                    NativeCmd::Seek(fraction),
                );
                repaint.request_repaint();
            })
        },
        on_volume: {
            let shared = Arc::clone(shared);
            let repaint = ctx.clone();
            Box::new(move |volume| {
                push_native_cmd(
                    &mut shared.native_cmds.lock().unwrap(),
                    NativeCmd::Volume(volume),
                );
                repaint.request_repaint();
            })
        },
        on_play: cmd(NativeCmd::Play),
        on_seek_back: cmd(NativeCmd::SeekBack),
        on_seek_forward: cmd(NativeCmd::SeekForward),
        on_mute: cmd(NativeCmd::Mute),
        on_fullscreen: cmd(NativeCmd::Fullscreen),
        on_save_frame: cmd(NativeCmd::SaveFrame),
        on_audio_track: {
            let shared = Arc::clone(shared);
            let repaint = ctx.clone();
            Box::new(move |track| {
                push_native_cmd(
                    &mut shared.native_cmds.lock().unwrap(),
                    NativeCmd::AudioTrack(track),
                );
                repaint.request_repaint();
            })
        },
        on_move: {
            let shared = Arc::clone(shared);
            let repaint = ctx.clone();
            Box::new(move |x, y| {
                *shared.native_panel_move.lock().unwrap() = Some((x, y));
                repaint.request_repaint();
            })
        },
    }
}

/// Natural-order comparison: runs of digits compare by numeric value, so
/// "img2" sorts before "img10". Inputs are expected to be already lowercased.
fn natural_cmp(a: &str, b: &str) -> CmpOrd {
    let mut ai = a.chars().peekable();
    let mut bi = b.chars().peekable();
    loop {
        match (ai.peek().copied(), bi.peek().copied()) {
            (None, None) => return CmpOrd::Equal,
            (None, Some(_)) => return CmpOrd::Less,
            (Some(_), None) => return CmpOrd::Greater,
            (Some(ca), Some(cb)) => {
                if ca.is_ascii_digit() && cb.is_ascii_digit() {
                    let da: String = take_while_digit(&mut ai);
                    let db: String = take_while_digit(&mut bi);
                    // Compare numeric value via trimmed-zero length then lexically,
                    // which avoids overflow on arbitrarily long digit runs.
                    let ta = da.trim_start_matches('0');
                    let tb = db.trim_start_matches('0');
                    let ord = ta.len().cmp(&tb.len()).then_with(|| ta.cmp(tb));
                    if ord != CmpOrd::Equal {
                        return ord;
                    }
                    // Equal value: fewer leading zeros sorts first for stability.
                    let ord = da.len().cmp(&db.len());
                    if ord != CmpOrd::Equal {
                        return ord;
                    }
                } else {
                    let ord = ca.cmp(&cb);
                    if ord != CmpOrd::Equal {
                        return ord;
                    }
                    ai.next();
                    bi.next();
                }
            }
        }
    }
}

fn take_while_digit(it: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut s = String::new();
    while let Some(&c) = it.peek() {
        if c.is_ascii_digit() {
            s.push(c);
            it.next();
        } else {
            break;
        }
    }
    s
}

fn is_image_file(path: &Path) -> bool {
    has_ext(path, IMAGE_EXTENSIONS)
}

fn video_key(item: &Item) -> (PathBuf, PathBuf) {
    (
        item.disk_path().to_path_buf(),
        item.media_name().to_path_buf(),
    )
}

fn is_media_name(path: &Path) -> bool {
    is_image_file(path) || is_video_file(path)
}

/// Sort in natural order (1, 2, ... 9, 10) rather than lexically (1, 10, 2).
fn sort_natural(items: Vec<Item>) -> Vec<Item> {
    let mut keyed: Vec<(String, Item)> = items.into_iter().map(|it| (it.sort_key(), it)).collect();
    keyed.sort_by(|a, b| natural_cmp(&a.0, &b.0));
    keyed.into_iter().map(|(_, it)| it).collect()
}

/// Walk the tree iteratively, descending into subfolders. Plain media files go
/// into `media`; archives and PDFs are only *recorded* in `containers`, not
/// opened — reading a container enumerates its entries but, on OneDrive, forces
/// the whole (often huge) file to download first, so that is deferred to a
/// background second phase. file_type() does not follow symlinks, so symlinked
/// directories are skipped and can't cause loops.
fn walk_directory(dir: &Path, media: &mut Vec<Item>, containers: &mut Vec<PathBuf>) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            match entry.file_type() {
                Ok(ft) if ft.is_dir() => stack.push(p),
                Ok(ft) if !ft.is_symlink() => {
                    if is_media_name(&p) {
                        media.push(Item::File(p));
                    } else if is_archive_file(&p) || is_pdf_file(&p) {
                        containers.push(p);
                    }
                }
                _ => {}
            }
        }
    }
}

/// Enumerate one archive's or PDF's browsable entries. This reads the file, so
/// on OneDrive it blocks until the placeholder has hydrated — run off the UI
/// thread, and only once the viewer navigates near the container.
fn expand_container(p: &Path) -> Vec<Item> {
    if is_archive_file(p) {
        scan_archive(p, is_media_name)
    } else if is_pdf_file(p) {
        scan_pdf(p)
    } else {
        Vec::new()
    }
}

/// The result of a completed folder scan: the browsing list and the root titles
/// are shown relative to. Produced on a background thread because even the
/// metadata-only tree walk can be slow on a large OneDrive tree.
struct ScanResult {
    root: PathBuf,
    items: Vec<Item>,
}

/// A container expansion delivered back to the UI: the archive/PDF that was
/// read and the entries it yielded (empty if unreadable/encrypted).
struct ExpandResult {
    path: PathBuf,
    entries: Vec<Item>,
}

/// Scan an opened selection into a browsing list. The tree is walked with
/// metadata only — archives and PDFs become single [`Item::Container`]
/// placeholders rather than being opened, so nothing downloads until the viewer
/// browses near it. A single opened directory is its own root; otherwise titles
/// are relative to the first path's parent.
fn run_scan(opened: Vec<PathBuf>, tx: Sender<ScanResult>, ctx: egui::Context) {
    let root = match opened.as_slice() {
        [] => {
            let _ = tx.send(ScanResult {
                root: PathBuf::new(),
                items: Vec::new(),
            });
            return;
        }
        [p] if p.is_dir() => p.clone(),
        [p, ..] => p.parent().unwrap_or(Path::new(".")).to_path_buf(),
    };

    let mut items = Vec::new();
    let mut containers = Vec::new();
    for p in opened {
        if p.is_dir() {
            walk_directory(&p, &mut items, &mut containers);
        } else if is_archive_file(&p) || is_pdf_file(&p) {
            containers.push(p);
        } else {
            items.push(Item::File(p));
        }
    }
    for c in containers {
        items.push(Item::Container { path: Arc::new(c) });
    }

    let _ = tx.send(ScanResult {
        root,
        items: sort_natural(items),
    });
    ctx.request_repaint();
}

fn circular_dist(idx: usize, cur: usize, count: usize) -> usize {
    if count == 0 {
        return usize::MAX;
    }
    let fwd = (idx + count - cur) % count;
    let bwd = (cur + count - idx) % count;
    fwd.min(bwd)
}

fn window_indices(cur: usize, count: usize, fwd: usize, bwd: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    let max = fwd.max(bwd).min(count.saturating_sub(1));
    for offset in 1..=max {
        if offset <= fwd {
            let idx = (cur + offset) % count;
            if seen.insert(idx) {
                result.push(idx);
            }
        }
        if offset <= bwd {
            let idx = (cur + count - offset) % count;
            if seen.insert(idx) {
                result.push(idx);
            }
        }
    }
    result
}

/// True while `idx` is still close enough to the on-screen image to be worth
/// working on; lets in-flight tasks bail when the user scrubs far away.
fn still_relevant(idx: usize, shared: &SharedState, max_dist: usize) -> bool {
    let cur = shared.current_idx.load(Ordering::Relaxed);
    let count = shared.image_count.load(Ordering::Relaxed);
    circular_dist(idx, cur, count) <= max_dist
}

/// Read the whole file and discard the bytes. This is purely the OneDrive "touch"
/// that forces the placeholder to download to local disk; it holds only a chunk
/// in memory at a time. Returns false if it bailed (drifted away or I/O error).
fn hydrate_file(path: &Path, idx: usize, shared: &SharedState, max_dist: usize) -> bool {
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = vec![0u8; READ_CHUNK];
    loop {
        if !still_relevant(idx, shared, max_dist) {
            return false;
        }
        match f.read(&mut buf) {
            Ok(0) => return true,
            Ok(_) => {} // discard
            Err(_) => return false,
        }
    }
}

/// Downscale so neither side exceeds `max_side`, preserving aspect ratio. Caps
/// both the GPU texture limit and our display budget, and keeps decode/upload
/// cheap for huge source images.
fn downscale_to_limit(w: u32, h: u32, rgba: Vec<u8>, max_side: u32) -> Result<(u32, u32, Vec<u8>)> {
    if w <= max_side && h <= max_side {
        return Ok((w, h, rgba));
    }
    let scale = (max_side as f64 / w as f64).min(max_side as f64 / h as f64);
    let nw = ((w as f64 * scale).floor() as u32).clamp(1, max_side);
    let nh = ((h as f64 * scale).floor() as u32).clamp(1, max_side);
    let buffer = image::RgbaImage::from_raw(w, h, rgba)
        .ok_or_else(|| anyhow::anyhow!("RGBA buffer does not match {}x{}", w, h))?;
    let resized = image::imageops::resize(&buffer, nw, nh, image::imageops::FilterType::Triangle);
    Ok((nw, nh, resized.into_raw()))
}

/// A fully decoded item ready for texture upload. `frames` holds RGBA frames
/// with their display durations in seconds: static images are a single frame;
/// animated GIFs carry every frame so the viewer can play them.
struct Decoded {
    width: u32,
    height: u32,
    frames: Vec<(Vec<u8>, f32)>,
}

fn decode_from_bytes(data: &[u8], max_side: u32) -> Result<Decoded> {
    // Detect the format from the file's magic bytes (more robust than the
    // extension) and decode to RGBA. `no_limits` lifts image's default 512MB
    // guard so large photos decode instead of being rejected as bombs; the
    // longest side is then capped by `downscale_to_limit` anyway.
    let mut reader = image::ImageReader::new(std::io::Cursor::new(data)).with_guessed_format()?;
    reader.no_limits();
    // GIFs decode to all their frames so animations play; if the animation
    // stream is broken, fall through and show the first frame as a still.
    if reader.format() == Some(image::ImageFormat::Gif)
        && let Ok(decoded) = decode_gif(data, max_side)
    {
        return Ok(decoded);
    }
    let rgba = reader.decode()?.into_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let (w, h, pixels) = downscale_to_limit(w, h, rgba.into_raw(), max_side)?;
    Ok(Decoded {
        width: w,
        height: h,
        frames: vec![(pixels, 0.0)],
    })
}

/// Turn a PDF page's extracted image into a decoded frame, capping the longest
/// side at `max_side` like `decode_from_bytes` does. A raw image arrives as
/// ready RGBA; an embedded JPEG still goes through the decoder.
fn pdf_decoded(img: PdfImage, max_side: u32) -> Option<Decoded> {
    match img {
        PdfImage::Rgba {
            width,
            height,
            data,
        } => {
            let (w, h, pixels) = downscale_to_limit(width, height, data, max_side).ok()?;
            Some(Decoded {
                width: w,
                height: h,
                frames: vec![(pixels, 0.0)],
            })
        }
        PdfImage::Encoded(bytes) => decode_from_bytes(&bytes, max_side).ok(),
    }
}

/// Total decoded-RGBA budget for one animation. A 60MB, 2080x2648, 48-frame
/// GIF is 1GB of full-size RGBA — decoding it whole exhausts memory, so frames
/// are downscaled to collectively fit this budget instead.
const ANIM_RGBA_BUDGET: u64 = 256 * 1024 * 1024;
/// Animations longer than this keep every k-th frame (screen time merged into
/// the kept neighbour) so the texture count stays sane.
const MAX_ANIM_FRAMES: u64 = 256;

/// Decode a GIF's frames (the decoder composites frame disposal for us),
/// streaming one frame at a time so peak memory is a single full-size frame
/// plus the budget-downscaled kept frames. A delay of 0 is the web's
/// "unspecified" convention; play it at the browser-standard 100ms.
fn decode_gif(data: &[u8], max_side: u32) -> Result<Decoded> {
    use image::AnimationDecoder;

    // Cheap pre-pass (skips pixel data): the canvas size and frame count pick
    // the per-frame downscale before any pixels are decoded.
    let mut opts = gif::DecodeOptions::new();
    opts.set_color_output(gif::ColorOutput::Indexed);
    let mut probe = opts.read_info(std::io::Cursor::new(data))?;
    let (cw, ch) = (probe.width() as u32, probe.height() as u32);
    let mut count: u64 = 0;
    while probe.next_frame_info()?.is_some() {
        count += 1;
    }
    if count == 0 {
        anyhow::bail!("GIF has no frames");
    }
    let step = count.div_ceil(MAX_ANIM_FRAMES);
    let kept = count.div_ceil(step);
    let budget_px = (ANIM_RGBA_BUDGET / 4 / kept) as f64;
    let scale = (budget_px / (cw as f64 * ch as f64)).sqrt().min(1.0);
    let side = max_side.min((cw.max(ch) as f64 * scale) as u32).max(1);

    let decoder = image::codecs::gif::GifDecoder::new(std::io::Cursor::new(data))?;
    let mut frames: Vec<(Vec<u8>, f32)> = Vec::with_capacity(kept as usize);
    let mut dims = (0, 0);
    for (i, frame) in decoder.into_frames().enumerate() {
        let frame = frame?;
        let (numer, denom) = frame.delay().numer_denom_ms();
        let ms = if denom == 0 {
            100.0
        } else {
            numer as f32 / denom as f32
        };
        let ms = if ms < 10.0 { 100.0 } else { ms };
        if !(i as u64).is_multiple_of(step) {
            // Dropped by decimation: its screen time stays on the kept frame.
            if let Some(last) = frames.last_mut() {
                last.1 += ms / 1000.0;
            }
            continue;
        }
        let buf = frame.into_buffer();
        let (w, h, pixels) = downscale_to_limit(buf.width(), buf.height(), buf.into_raw(), side)?;
        if frames.is_empty() {
            dims = (w, h);
        } else if (w, h) != dims {
            anyhow::bail!("GIF frames change size");
        }
        frames.push((pixels, ms / 1000.0));
    }
    if frames.is_empty() {
        anyhow::bail!("GIF has no frames");
    }
    Ok(Decoded {
        width: dims.0,
        height: dims.1,
        frames,
    })
}

fn spawn_hydrate(
    pool: &ThreadPool,
    idx: usize,
    path: PathBuf,
    shared: Arc<SharedState>,
    done_tx: Sender<(PathBuf, bool)>,
) {
    let max_dist = HYDRATE_AHEAD.max(HYDRATE_BEHIND);
    pool.spawn(move || {
        let ok = hydrate_file(&path, idx, &shared, max_dist);
        let _ = done_tx.send((path, ok));
    });
}

#[allow(clippy::too_many_arguments)]
fn spawn_decode(
    pool: &ThreadPool,
    idx: usize,
    item: Item,
    max_side: u32,
    shared: Arc<SharedState>,
    max_dist: usize,
    decoded_tx: Sender<(usize, Decoded)>,
    done_tx: Sender<(usize, DecodeOutcome)>,
    ctx: egui::Context,
) {
    pool.spawn(move || {
        let cancelled = || !still_relevant(idx, &shared, max_dist);
        // PDF pages are decoded straight from the resident document — raw images
        // arrive as ready RGBA, embedded JPEGs go through the shared decoder —
        // so they never touch the file-bytes path or its re-encode round trip.
        let outcome = if item.is_pdf() {
            if cancelled() {
                DecodeOutcome::Aborted
            } else {
                match item.pdf_image().and_then(|img| pdf_decoded(img, max_side)) {
                    Some(decoded) => {
                        let _ = decoded_tx.send((idx, decoded));
                        DecodeOutcome::Ok
                    }
                    None => DecodeOutcome::Failed,
                }
            }
        } else {
            match item.read(&cancelled) {
                Some(data) => match decode_from_bytes(&data, max_side) {
                    Ok(decoded) => {
                        let _ = decoded_tx.send((idx, decoded));
                        DecodeOutcome::Ok
                    }
                    Err(_) => DecodeOutcome::Failed,
                },
                // `read` bails on both a stale index and an I/O/archive error;
                // only a real error counts against the retry budget.
                None if cancelled() => DecodeOutcome::Aborted,
                None => DecodeOutcome::Failed,
            }
        };
        let _ = done_tx.send((idx, outcome));
        ctx.request_repaint();
    });
}

/// Bookkeeping for the coordinator's in-flight and completed work. Bundling it
/// lets the two completion-draining sites (non-blocking sweep and the blocking
/// `select!`) share one apply path instead of duplicating it.
#[derive(Default)]
struct CoordState {
    // Disk files already pulled local this session, keyed by path so an archive
    // counts once no matter how many entries it contributes. Persisted (never
    // evicted) so we never re-download or re-read a 23MB file we've touched.
    hydrated: HashSet<PathBuf>,
    hy_in_flight: HashSet<PathBuf>,
    dc_in_flight: HashSet<usize>,
    decoded_sent: HashSet<usize>,
    failed: HashMap<usize, usize>,
}

impl CoordState {
    fn apply_hydrate(&mut self, path: PathBuf, ok: bool) {
        self.hy_in_flight.remove(&path);
        if ok {
            self.hydrated.insert(path);
        }
    }

    fn apply_decode(&mut self, idx: usize, outcome: DecodeOutcome, disk: &Path) {
        self.dc_in_flight.remove(&idx);
        match outcome {
            DecodeOutcome::Ok => {
                self.decoded_sent.insert(idx);
                self.failed.remove(&idx);
                self.hydrated.insert(disk.to_path_buf());
            }
            DecodeOutcome::Failed => {
                *self.failed.entry(idx).or_insert(0) += 1;
            }
            DecodeOutcome::Aborted => {}
        }
    }
}

fn run_coordinator(
    items: Arc<Vec<Item>>,
    shared: Arc<SharedState>,
    decoded_tx: Sender<(usize, Decoded)>,
    repaint_ctx: egui::Context,
    max_side: u32,
) {
    let count = items.len();
    if count == 0 {
        return;
    }

    let hydrate_pool = ThreadPool::new(HYDRATE_WORKERS, "Kagami-hydrate");
    let decode_pool = ThreadPool::new(DECODE_WORKERS, "Kagami-decode");
    let priority_pool = ThreadPool::new(PRIORITY_WORKERS, "Kagami-priority");

    let (hy_done_tx, hy_done_rx) = crossbeam_channel::unbounded::<(PathBuf, bool)>();
    let (dc_done_tx, dc_done_rx) = crossbeam_channel::unbounded::<(usize, DecodeOutcome)>();

    let mut state = CoordState::default();

    let dc_max = DECODE_AHEAD.max(DECODE_BEHIND);
    let max_dc_in_flight = DECODE_WORKERS + PRIORITY_WORKERS;

    loop {
        // A newer pipeline (another folder opened, or this scan's second phase
        // installing the full list) supersedes us: stop rather than keep the
        // pools and this loop alive forever on a stale list.
        if shared.cancelled.load(Ordering::Relaxed) {
            return;
        }

        // Drain completed work.
        while let Ok((path, ok)) = hy_done_rx.try_recv() {
            state.apply_hydrate(path, ok);
        }
        while let Ok((idx, outcome)) = dc_done_rx.try_recv() {
            state.apply_decode(idx, outcome, items[idx].disk_path());
        }

        let cur = shared.current_idx.load(Ordering::Relaxed);

        // Videos are decoded/played by the on-screen VideoPlayer, not here, so
        // treat a current video as "ready" to let look-ahead for images proceed.
        // An un-expanded archive/PDF placeholder has nothing to decode either
        // (the UI reads it on approach), so treat it as ready too.
        let current_is_video = is_video_file(items[cur].media_name());
        let current_is_container = items[cur].is_container();

        // Highest priority: get the on-screen image decoded. It runs on its own
        // pool so it never waits behind look-ahead work, and it gets first claim
        // on download bandwidth because nothing else is scheduled until it lands.
        let current_ready =
            current_is_video || current_is_container || state.decoded_sent.contains(&cur);
        if !current_ready
            && !state.dc_in_flight.contains(&cur)
            && *state.failed.get(&cur).unwrap_or(&0) < MAX_DECODE_RETRIES
        {
            state.dc_in_flight.insert(cur);
            spawn_decode(
                &priority_pool,
                cur,
                items[cur].clone(),
                max_side,
                shared.clone(),
                dc_max,
                decoded_tx.clone(),
                dc_done_tx.clone(),
                repaint_ctx.clone(),
            );
        }

        // Only fan out look-ahead once the current image is showing, so it never
        // competes with the visible image for OneDrive bandwidth or CPU.
        if current_ready {
            // Decode the immediate neighbours so adjacent navigation is instant.
            for idx in window_indices(cur, count, DECODE_AHEAD, DECODE_BEHIND) {
                if state.dc_in_flight.len() >= max_dc_in_flight {
                    break;
                }
                if !is_image_file(items[idx].media_name()) {
                    continue; // videos are handled on demand by the player
                }
                if state.decoded_sent.contains(&idx) || state.dc_in_flight.contains(&idx) {
                    continue;
                }
                if *state.failed.get(&idx).unwrap_or(&0) >= MAX_DECODE_RETRIES {
                    continue;
                }
                state.dc_in_flight.insert(idx);
                spawn_decode(
                    &decode_pool,
                    idx,
                    items[idx].clone(),
                    max_side,
                    shared.clone(),
                    dc_max,
                    decoded_tx.clone(),
                    dc_done_tx.clone(),
                    repaint_ctx.clone(),
                );
            }

            // Pre-hydrate a wide window from the cloud so future navigation reads
            // from local disk instead of waiting on a download.
            for idx in window_indices(cur, count, HYDRATE_AHEAD, HYDRATE_BEHIND) {
                if state.hy_in_flight.len() >= HYDRATE_WORKERS {
                    break;
                }
                // Don't pre-download videos: they can be huge and the player
                // streams them on demand when navigated to.
                if !is_image_file(items[idx].media_name()) {
                    continue;
                }
                let target = items[idx].disk_path();
                if state.hydrated.contains(target)
                    || state.hy_in_flight.contains(target)
                    || state.dc_in_flight.contains(&idx)
                {
                    continue;
                }
                state.hy_in_flight.insert(target.to_path_buf());
                spawn_hydrate(
                    &hydrate_pool,
                    idx,
                    target.to_path_buf(),
                    shared.clone(),
                    hy_done_tx.clone(),
                );
            }
        }

        // Forget decode bookkeeping outside the decode window (textures are
        // pruned on the UI side). `hydrated` is intentionally kept forever.
        state
            .decoded_sent
            .retain(|&idx| circular_dist(idx, cur, count) <= dc_max);
        state
            .failed
            .retain(|&idx, _| circular_dist(idx, cur, count) <= dc_max);

        // Wake promptly while the current image is still pending, idle otherwise.
        let timeout = if current_ready {
            Duration::from_millis(20)
        } else {
            Duration::from_millis(2)
        };
        crossbeam_channel::select! {
            recv(hy_done_rx) -> msg => {
                if let Ok((path, ok)) = msg {
                    state.apply_hydrate(path, ok);
                }
            }
            recv(dc_done_rx) -> msg => {
                if let Ok((idx, outcome)) = msg {
                    state.apply_decode(idx, outcome, items[idx].disk_path());
                }
            }
            default(timeout) => {}
        }
    }
}

/// A fresh ~/Downloads/<video-name>_<timestamp>.png path for a saved frame.
fn downloads_png_path(src: &Path) -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    Some(home.join("Downloads").join(format!("{stem}_{ts}.png")))
}

#[cfg(not(target_os = "macos"))]
struct VideoOverlay {
    position: f64,
    duration: f64,
    paused: bool,
    muted: bool,
    volume: f32,
}

fn fmt_time(t: f64) -> String {
    if !t.is_finite() || t < 0.0 {
        return "0:00".to_string();
    }
    let total = t as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// Height of the bottom strip that holds the video controls.
#[cfg(not(target_os = "macos"))]
const CONTROLS_STRIP_H: f32 = 96.0;

/// Paint the control strip: a gradient backdrop, the progress bar, the
/// time/volume status and a one-line key hint. Pure drawing — interaction is
/// handled by the caller's single response. Returns the bar's geometry so the
/// caller can map a click/drag position to a seek time. `scrubbing` enlarges
/// the playhead for feedback.
#[cfg(not(target_os = "macos"))]
fn draw_video_overlay(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    info: &VideoOverlay,
    scrubbing: bool,
) -> egui::Rect {
    let margin = 28.0;
    let bar_h = 5.0;
    let bar = egui::Rect::from_min_max(
        egui::pos2(rect.left() + margin, rect.bottom() - 34.0 - bar_h),
        egui::pos2(rect.right() - margin, rect.bottom() - 34.0),
    );

    let painter = ui.painter();

    // Bottom-up gradient so the white text/bar stay readable over any frame.
    let mut mesh = egui::Mesh::default();
    let top = rect.bottom() - CONTROLS_STRIP_H;
    let clear = egui::Color32::from_black_alpha(0);
    let dark = egui::Color32::from_black_alpha(170);
    mesh.colored_vertex(egui::pos2(rect.left(), top), clear);
    mesh.colored_vertex(egui::pos2(rect.right(), top), clear);
    mesh.colored_vertex(egui::pos2(rect.left(), rect.bottom()), dark);
    mesh.colored_vertex(egui::pos2(rect.right(), rect.bottom()), dark);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(1, 2, 3);
    painter.add(mesh);

    let frac = if info.duration > 0.0 {
        (info.position / info.duration).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    painter.rect_filled(bar, 2.5, egui::Color32::from_white_alpha(60));
    let mut fill = bar;
    fill.set_right(bar.left() + bar.width() * frac);
    painter.rect_filled(fill, 2.5, egui::Color32::from_rgb(240, 240, 240));
    let knob = if scrubbing { 8.0 } else { 6.0 };
    painter.circle_filled(
        egui::pos2(fill.right(), bar.center().y),
        knob,
        egui::Color32::WHITE,
    );

    let vol = if info.muted || info.volume <= 0.0 {
        "muted".to_string()
    } else {
        format!("vol {}%", (info.volume * 100.0).round() as i32)
    };
    let status = format!(
        "{}  {} / {}      {}",
        if info.paused { "II" } else { ">" },
        fmt_time(info.position),
        fmt_time(info.duration),
        vol,
    );
    painter.text(
        egui::pos2(bar.left(), bar.top() - 10.0),
        egui::Align2::LEFT_BOTTOM,
        status,
        egui::FontId::proportional(15.0),
        egui::Color32::WHITE,
    );
    painter.text(
        egui::pos2(bar.right(), bar.top() - 10.0),
        egui::Align2::RIGHT_BOTTOM,
        "Space play/pause   Up/Down seek 5s   M mute   Left/Right file   N open   Cmd+Del trash",
        egui::FontId::proportional(13.0),
        egui::Color32::from_white_alpha(160),
    );

    bar
}

/// An uploaded item: one texture per frame (stills have exactly one) plus the
/// per-frame durations in seconds that drive GIF animation.
struct CachedImage {
    frames: Vec<egui::TextureHandle>,
    delays: Vec<f32>,
}

impl CachedImage {
    /// Frame to draw at absolute time `t` and the seconds until the next
    /// frame change (infinite for stills, so no repaint gets scheduled).
    fn frame_at(&self, t: f64) -> (usize, f32) {
        let total: f32 = self.delays.iter().sum();
        if self.delays.len() < 2 || total <= 0.0 {
            return (0, f32::INFINITY);
        }
        let mut t = (t % total as f64) as f32;
        for (i, d) in self.delays.iter().enumerate() {
            if t < *d {
                return (i, d - t);
            }
            t -= d;
        }
        (0, self.delays[0])
    }
}

struct KagamiApp {
    /// Everything being browsed: plain files and archive entries, pre-sorted.
    /// Shared (by `Arc`) with the background coordinator, so opening a folder
    /// hands over a refcount bump rather than cloning the whole list.
    items: Arc<Vec<Item>>,
    /// The opened folder. Titles show each image's path relative to it, so an
    /// image in a subfolder is shown as `sub/img.jpg`.
    root: PathBuf,
    current_index: usize,
    cache: HashMap<usize, CachedImage>,
    decoded_rx: Receiver<(usize, Decoded)>,
    /// Receives the result of the current off-thread folder scan, if one is
    /// running. Replacing it (a newer open) drops this receiver so a stale
    /// scan's result is ignored.
    scan_rx: Option<Receiver<ScanResult>>,
    /// True while a scan is in flight and no pipeline is installed yet, so the
    /// empty view shows "Scanning…" rather than "No images found".
    scanning: bool,
    /// Archive/PDF expansions delivered from background reader threads, folded
    /// into the list in place of their placeholder as they arrive.
    expand_rx: Receiver<ExpandResult>,
    expand_tx: Sender<ExpandResult>,
    /// Container paths currently being read, so a placeholder is only expanded
    /// once even while its read is still in flight.
    expanding: HashSet<PathBuf>,
    shared: Arc<SharedState>,
    last_title_index: Option<usize>,
    max_side: u32,
    /// Image zoom factor relative to fit-to-window (1.0 = fitted). Pan is the
    /// screen-pixel offset of the image centre from the viewport centre. Both
    /// reset on navigation and apply only to images, never video.
    zoom: f32,
    pan: egui::Vec2,
    /// View rotation in clockwise quarter-turns (0..4): R turns right, E turns
    /// left, T resets.
    /// Display-only — the file is untouched — and reset on navigation.
    rotation: u8,
    /// The currently on-screen video and its index, if the current item is a
    /// video. Dropping it stops playback and tears down the audio stream.
    video: Option<(usize, VideoPlayer)>,
    video_host: Option<video::VideoHost>,
    subtitle_scale: i64,
    video_error: Option<String>,
    /// Last playback position per video, keyed by (backing file, entry name),
    /// so navigating away and back resumes where the video left off.
    video_positions: HashMap<(PathBuf, PathBuf), f64>,
    /// The native AppKit control bar, created lazily on first video playback and
    /// reused (just hidden) thereafter.
    #[cfg(target_os = "macos")]
    native_controls: Option<controls::NativeControls>,
    /// egui time until which the video controls stay visible; bumped on pointer
    /// movement or a control key so they fade out only while idle during play.
    controls_until: f64,
    /// IINA-style normalized floating-OSC position. `None` uses IINA's defaults
    /// (horizontal 0.5, vertical 0.1).
    playback_panel_pos: Option<PanelPosition>,
    /// Native fullscreen state we track ourselves (eframe's viewport flag is
    /// often `None`, which made the toggle misfire).
    fullscreen: bool,
    /// Indices trashed this session. Kept (rather than removed from `paths`) so
    /// the background coordinator's stable indices stay valid; navigation skips
    /// them and they read as gaps.
    deleted: HashSet<usize>,
    /// egui context kept so a folder opened later can spawn its own coordinator
    /// with a fresh repaint handle.
    egui_ctx: egui::Context,
    /// Keep activating the app and focusing the window until this deadline.
    /// The launch picker is a modal panel; when it closes, macOS hands
    /// activation back to the previously active app asynchronously, which can
    /// land after a fixed number of frames — so re-assert on a time budget.
    focus_until: Option<std::time::Instant>,
    first_ui_at: Option<std::time::Instant>,
}

impl KagamiApp {
    fn new(cc: &eframe::CreationContext<'_>, initial: Vec<PathBuf>) -> Self {
        #[cfg(target_os = "macos")]
        open_docs::set_repaint(cc.egui_ctx.clone());
        // Cap textures only at the GPU's max 2D dimension so uploads never panic;
        // images are otherwise decoded at their full resolution.
        let max_side = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| rs.device.limits().max_texture_dimension_2d)
            .unwrap_or(8192)
            .max(1);

        // A placeholder pipeline; `open_paths` installs the real one (now, if a
        // path was passed, or after the startup picker on the first frame).
        let (_tx, decoded_rx) = crossbeam_channel::unbounded();
        let (expand_tx, expand_rx) = crossbeam_channel::unbounded();
        let mut app = Self {
            items: Arc::new(Vec::new()),
            root: PathBuf::new(),
            current_index: 0,
            cache: HashMap::new(),
            decoded_rx,
            scan_rx: None,
            scanning: false,
            expand_rx,
            expand_tx,
            expanding: HashSet::new(),
            shared: Arc::new(SharedState {
                current_idx: AtomicUsize::new(0),
                image_count: AtomicUsize::new(0),
                cancelled: AtomicBool::new(false),
                #[cfg(target_os = "macos")]
                native_cmds: Mutex::new(Vec::new()),
                #[cfg(target_os = "macos")]
                native_panel_move: Mutex::new(None),
            }),
            last_title_index: None,
            max_side,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            rotation: 0,
            video: None,
            video_host: video::VideoHost::from_window(cc),
            subtitle_scale: video::SUBTITLE_SCALE_DEFAULT,
            video_error: None,
            video_positions: HashMap::new(),
            #[cfg(target_os = "macos")]
            native_controls: None,
            controls_until: 0.0,
            playback_panel_pos: cc.storage.and_then(|s| eframe::get_value(s, PANEL_POS_KEY)),
            fullscreen: false,
            deleted: HashSet::new(),
            egui_ctx: cc.egui_ctx.clone(),
            focus_until: None,
            first_ui_at: None,
        };
        app.request_focus(800);
        app.open_paths(initial);
        app
    }

    /// Keep re-asserting app activation and window focus for `ms` after a modal
    /// (the launch/open picker) steals it, since macOS hands activation back
    /// asynchronously. See `focus_until`.
    fn request_focus(&mut self, ms: u64) {
        self.focus_until = Some(std::time::Instant::now() + Duration::from_millis(ms));
    }

    /// Open a selection of paths. The scan (which walks subfolders and, on
    /// OneDrive, can block on network hydration of placeholder directories)
    /// runs on a background thread so the window never freezes; the real
    /// pipeline is installed by `install_scanned` when the result arrives.
    /// Until then the view shows "Scanning…".
    fn open_paths(&mut self, opened: Vec<PathBuf>) {
        if opened.is_empty() {
            return;
        }
        // Scan off the UI thread (the tree walk can be slow on a big OneDrive
        // tree). Replacing `scan_rx` drops any prior receiver, so a still-running
        // older scan's result is discarded.
        let (tx, rx) = crossbeam_channel::unbounded();
        let ctx = self.egui_ctx.clone();
        thread::spawn(move || run_scan(opened, tx, ctx));
        self.scan_rx = Some(rx);
        self.scanning = true;
        // Tear down the current pipeline so the stale folder stops showing while
        // the new scan runs.
        self.tear_down();
    }

    /// Clear the browsing state to an empty pipeline. Used while a scan is in
    /// flight so no stale folder shows.
    fn tear_down(&mut self) {
        self.root = PathBuf::new();
        self.cache.clear();
        self.deleted.clear();
        self.expanding.clear();
        self.video = None;
        self.video_error = None;
        self.video_positions.clear();
        self.last_title_index = None;
        self.rotation = 0;
        self.reset_zoom();
        self.start_coordinator(Arc::new(Vec::new()), 0);
    }

    /// Install the browsing list from a completed scan and start the background
    /// coordinator.
    fn install_scanned(&mut self, result: ScanResult) {
        let ScanResult { root, items } = result;
        self.root = root;
        self.cache.clear();
        self.deleted.clear();
        self.expanding.clear();
        self.video = None;
        self.video_error = None;
        self.video_positions.clear();
        self.last_title_index = None;
        self.rotation = 0;
        self.reset_zoom();
        self.start_coordinator(Arc::new(items), 0);
    }

    /// Point the pipeline at `items` with the view on `current_index`, spawning
    /// a fresh coordinator and cancelling the previous one (which otherwise
    /// loops forever). Callers set up `root`/`cache`/etc. around this.
    fn start_coordinator(&mut self, items: Arc<Vec<Item>>, current_index: usize) {
        self.shared.cancelled.store(true, Ordering::Relaxed);
        let shared = Arc::new(SharedState {
            current_idx: AtomicUsize::new(current_index),
            image_count: AtomicUsize::new(items.len()),
            cancelled: AtomicBool::new(false),
            #[cfg(target_os = "macos")]
            native_cmds: Mutex::new(Vec::new()),
            #[cfg(target_os = "macos")]
            native_panel_move: Mutex::new(None),
        });
        let (decoded_tx, decoded_rx) = crossbeam_channel::unbounded();
        let items_arc = items.clone();
        let shared2 = shared.clone();
        let ctx = self.egui_ctx.clone();
        let max_side = self.max_side;
        thread::spawn(move || run_coordinator(items_arc, shared2, decoded_tx, ctx, max_side));

        self.items = items;
        self.current_index = current_index;
        self.shared = shared;
        #[cfg(target_os = "macos")]
        if let Some(controls) = &self.native_controls {
            controls.set_callbacks(native_controls_callbacks(&self.shared, &self.egui_ctx));
        }
        self.decoded_rx = decoded_rx;
    }

    /// Kick off reading any un-expanded archive/PDF whose placeholder is within
    /// a small window of the current view, so browsing near it (and only then)
    /// downloads it. Runs each frame; the `expanding` set keeps it to one read
    /// per container.
    fn maybe_expand(&mut self) {
        let count = self.items.len();
        if count == 0 {
            return;
        }
        let cur = self.current_index;
        let mut idxs = window_indices(cur, count, EXPAND_AHEAD, EXPAND_BEHIND);
        idxs.push(cur);
        for i in idxs {
            let Item::Container { path } = &self.items[i] else {
                continue;
            };
            let path = path.as_ref().clone();
            if !self.expanding.insert(path.clone()) {
                continue; // already reading this one
            }
            let tx = self.expand_tx.clone();
            let ctx = self.egui_ctx.clone();
            thread::spawn(move || {
                let entries = expand_container(&path);
                let _ = tx.send(ExpandResult { path, entries });
                ctx.request_repaint();
            });
        }
    }

    /// Fold a finished container expansion into the list in place of its
    /// placeholder, keeping the on-screen item (and decoded textures) in view.
    fn apply_expansion(&mut self, res: ExpandResult) {
        self.expanding.remove(&res.path);
        let old = self.items.clone();
        let cpath = res.path.as_path();

        // Was the placeholder the item on screen? If so, land on the first page
        // of the archive/PDF just opened rather than jumping to the start.
        let on_placeholder = matches!(
            old.get(self.current_index),
            Some(Item::Container { path }) if path.as_path() == cpath
        );

        let mut next: Vec<Item> = old
            .iter()
            .filter(|it| !matches!(it, Item::Container { path } if path.as_path() == cpath))
            .cloned()
            .collect();
        next.extend(res.entries);
        let next = sort_natural(next);

        // Map indexed state across the reordered list by item identity.
        let mut key_to_new: HashMap<String, usize> = HashMap::with_capacity(next.len());
        for (i, it) in next.iter().enumerate() {
            key_to_new.entry(it.sort_key()).or_insert(i);
        }
        let remap = |old_idx: usize| -> Option<usize> {
            old.get(old_idx)
                .and_then(|it| key_to_new.get(&it.sort_key()).copied())
        };

        let current_index = if on_placeholder {
            next.iter()
                .position(|it| it.disk_path() == cpath)
                .unwrap_or(0)
        } else {
            remap(self.current_index).unwrap_or(0)
        };

        // Preserve decoded textures, deletions and any open video across the
        // reindex so the view doesn't flicker or lose its trashed set.
        let mut old_cache = std::mem::take(&mut self.cache);
        let mut new_cache = HashMap::with_capacity(old_cache.len());
        for (old_idx, img) in old_cache.drain() {
            if let Some(n) = remap(old_idx) {
                new_cache.insert(n, img);
            }
        }
        self.cache = new_cache;
        self.deleted = self.deleted.iter().filter_map(|&i| remap(i)).collect();
        self.video = self
            .video
            .take()
            .and_then(|(idx, p)| remap(idx).map(|n| (n, p)));
        self.last_title_index = None;

        self.start_coordinator(Arc::new(next), current_index);
    }

    /// Nearest non-deleted index from `start` in direction `dir` (+1 / -1),
    /// wrapping around. None if every file is deleted.
    fn neighbor(&self, start: usize, dir: isize) -> Option<usize> {
        let n = self.items.len();
        if n == 0 || self.deleted.len() >= n {
            return None;
        }
        let mut i = start;
        for _ in 0..n {
            i = if dir > 0 {
                (i + 1) % n
            } else {
                (i + n - 1) % n
            };
            if !self.deleted.contains(&i) {
                return Some(i);
            }
        }
        None
    }

    /// Move the current file to the Trash, then advance to a remaining one.
    /// Archive entries and PDF pages are read-only views: trashing one would
    /// take the whole container (and every other entry) with it, so refuse.
    fn delete_current(&mut self) {
        let cur = self.current_index;
        if self.items.is_empty() || self.deleted.contains(&cur) {
            return;
        }
        if self.items[cur].is_archived() || self.items[cur].is_pdf() {
            eprintln!(
                "[delete] {} is inside a container; not trashing",
                self.items[cur].display(&self.root)
            );
            return;
        }
        if self.items[cur].is_container() {
            // Still opening; nothing meaningful to trash yet.
            return;
        }
        let path = self.items[cur].disk_path();
        if let Err(e) = trash::delete(path) {
            eprintln!("[delete] could not trash {path:?}: {e}");
            return;
        }
        self.deleted.insert(cur);
        self.cache.remove(&cur);
        if matches!(&self.video, Some((i, _)) if *i == cur) {
            self.video = None;
            self.video_error = None;
        }
        if let Some(next) = self.neighbor(cur, 1) {
            self.current_index = next;
            self.shared.current_idx.store(next, Ordering::Relaxed);
            self.prune_cache();
            self.reset_zoom();
        }
    }

    fn reset_zoom(&mut self) {
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
    }

    /// Draw the current image, applying zoom (pinch gesture / ctrl-scroll / the
    /// `+` `-` keys) and pan (drag or scroll while zoomed). Zoom is centred on
    /// the cursor for gestures and on the viewport for keys; pan is clamped so
    /// the image can't be dragged off the viewport.
    fn show_image(&mut self, ui: &mut egui::Ui, rect: egui::Rect, texture: &egui::TextureHandle) {
        const MAX_ZOOM: f32 = 32.0;
        const KEY_STEP: f32 = 1.25;

        // R turns the view a quarter clockwise, E a quarter counterclockwise,
        // T resets; an odd rotation swaps the aspect the fit/pan math sees,
        // while the rotated UVs below do the actual turning at draw time.
        let (r_key, e_key, t_key, pinch, scroll, pointer, key_in, key_out, key_reset) =
            ui.input(|i| {
                (
                    i.key_pressed(egui::Key::R),
                    i.key_pressed(egui::Key::E),
                    i.key_pressed(egui::Key::T),
                    i.zoom_delta(),
                    i.smooth_scroll_delta,
                    i.pointer.hover_pos(),
                    i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals),
                    i.key_pressed(egui::Key::Minus),
                    i.key_pressed(egui::Key::Num0),
                )
            });
        if r_key {
            self.rotation = (self.rotation + 1) % 4;
        }
        if e_key {
            self.rotation = (self.rotation + 3) % 4;
        }
        if t_key {
            self.rotation = 0;
        }
        let tex_size = texture.size_vec2();
        let img_size = if self.rotation % 2 == 1 {
            egui::vec2(tex_size.y, tex_size.x)
        } else {
            tex_size
        };
        let fit = (rect.width() / img_size.x).min(rect.height() / img_size.y);

        if key_reset {
            self.reset_zoom();
        }

        // Trackpad pinch (and ctrl/⌘-scroll) zoom about the cursor; keys zoom
        // about the viewport centre. Both keep the focal point fixed on screen.
        let mut new_zoom = self.zoom;
        let mut focus = rect.center();
        if pinch != 1.0 {
            new_zoom *= pinch;
            focus = pointer.unwrap_or(focus);
        }
        if key_in {
            new_zoom *= KEY_STEP;
        }
        if key_out {
            new_zoom /= KEY_STEP;
        }
        new_zoom = new_zoom.clamp(1.0, MAX_ZOOM);
        if new_zoom != self.zoom {
            let f = new_zoom / self.zoom;
            self.pan = (focus - rect.center()) * (1.0 - f) + self.pan * f;
            self.zoom = new_zoom;
        }

        // Pan by dragging or scrolling, but only when zoomed past fit.
        let resp = ui.interact(
            rect,
            ui.id().with("image_pan"),
            egui::Sense::click_and_drag(),
        );
        if self.zoom > 1.0 {
            self.pan += resp.drag_delta() + scroll;
            ui.ctx().set_cursor_icon(if resp.dragged() {
                egui::CursorIcon::Grabbing
            } else {
                egui::CursorIcon::Grab
            });
        }
        if self.zoom <= 1.0 {
            self.pan = egui::Vec2::ZERO;
        }

        // Clamp pan so a panned edge never crosses into the viewport; centre any
        // axis where the scaled image is smaller than the viewport.
        let scaled = img_size * (fit * self.zoom);
        let clamp = |img: f32, view: f32, p: f32| {
            let slack = (img - view) / 2.0;
            if slack > 0.0 {
                p.clamp(-slack, slack)
            } else {
                0.0
            }
        };
        self.pan.x = clamp(scaled.x, rect.width(), self.pan.x);
        self.pan.y = clamp(scaled.y, rect.height(), self.pan.y);

        let img_rect = egui::Rect::from_center_size(rect.center() + self.pan, scaled);
        let painter = ui.painter().with_clip_rect(rect);
        if self.rotation == 0 {
            painter.image(
                texture.id(),
                img_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        } else {
            // Quarter-turn by walking the UV corners backwards around the
            // quad: screen corner i samples texture corner (i - turns) mod 4
            // (both enumerated clockwise from top-left).
            let corners = [
                img_rect.left_top(),
                img_rect.right_top(),
                img_rect.right_bottom(),
                img_rect.left_bottom(),
            ];
            let uvs = [
                egui::pos2(0.0, 0.0),
                egui::pos2(1.0, 0.0),
                egui::pos2(1.0, 1.0),
                egui::pos2(0.0, 1.0),
            ];
            let mut mesh = egui::Mesh::with_texture(texture.id());
            for (i, pos) in corners.into_iter().enumerate() {
                mesh.vertices.push(egui::epaint::Vertex {
                    pos,
                    uv: uvs[(i + 4 - self.rotation as usize) % 4],
                    color: egui::Color32::WHITE,
                });
            }
            mesh.add_triangle(0, 1, 2);
            mesh.add_triangle(0, 2, 3);
            painter.add(mesh);
        }
    }

    /// Native macOS fullscreen. The video is a CALayer-backed NSView, so the
    /// system animates it smoothly (the whole point of the native render path).
    fn set_fullscreen(&mut self, ctx: &egui::Context, on: bool) {
        if self.fullscreen == on {
            return;
        }
        self.fullscreen = on;
        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(on));
    }

    /// Make sure the video player matches the current item: tear down any player
    /// for a different (or non-video) item, and open one for a current video.
    fn sync_video(&mut self, ctx: &egui::Context) {
        let cur = self.current_index;
        let is_video = is_video_file(self.items[cur].media_name());

        if let Some((idx, player)) = &self.video
            && (*idx != cur || !is_video)
        {
            self.video_positions
                .insert(video_key(&self.items[*idx]), player.position());
            self.video = None;
            self.video_error = None;
        }
        if is_video && self.video.is_none() && !video::libvlc_ready() {
            ctx.request_repaint_after(Duration::from_millis(50));
            return;
        }
        if is_video && self.video.is_none() {
            // libvlc renders natively into its own NSView over the window.
            // Archived videos play in place: stored entries straight from the
            // archive file, compressed ones from a decompressed buffer.
            let opened = match &self.video_host {
                None => Err(anyhow::anyhow!("no window to attach the video view to")),
                Some(host) => match self.items[cur].video_source() {
                    Some(VideoSource::Path(p)) => VideoPlayer::open(&p, host),
                    Some(VideoSource::FileRange { path, start, len }) => {
                        VideoPlayer::open_range(&path, start, len, host)
                    }
                    Some(VideoSource::Bytes(data)) => VideoPlayer::open_bytes(data, host),
                    None => Err(anyhow::anyhow!("could not read the archive entry")),
                },
            };
            match opened {
                Ok(mut player) => {
                    player.set_subtitle_scale(self.subtitle_scale);
                    if let Some(&pos) = self.video_positions.get(&video_key(&self.items[cur]))
                        && pos > 1.0
                    {
                        player.resume_from(pos);
                    }
                    self.video = Some((cur, player));
                    self.video_error = None;
                }
                Err(e) => self.video_error = Some(format!("Cannot play video: {e}")),
            }
        }
    }

    /// Drain pending native-panel commands (for the caller to apply to the
    /// player) and fold a pending panel move into `playback_panel_pos`. Reads
    /// from the shared `Arc` so it doesn't conflict with the live `&mut player`.
    #[cfg(target_os = "macos")]
    fn drain_native(&mut self) -> Vec<NativeCmd> {
        let cmds = std::mem::take(&mut *self.shared.native_cmds.lock().unwrap());
        if let Some((horizontal, vertical)) = self.shared.native_panel_move.lock().unwrap().take() {
            self.playback_panel_pos = Some(PanelPosition {
                horizontal: horizontal as f32,
                vertical: vertical as f32,
            });
        }
        cmds
    }

    fn prune_cache(&mut self) {
        if self.items.is_empty() {
            return;
        }
        let count = self.items.len();
        let cur = self.current_index;
        let max = DECODE_AHEAD.max(DECODE_BEHIND) + 1;
        self.cache
            .retain(|&idx, _| circular_dist(idx, cur, count) <= max);
    }

    fn update_title(&mut self, ctx: &egui::Context) {
        if self.last_title_index == Some(self.current_index) {
            return;
        }
        // Show the path relative to the opened folder, so images in a subfolder
        // read as `sub/img.jpg` and archive entries as `comics.cbz/page01.jpg`.
        let name = self.items[self.current_index].display(&self.root);
        // Count by non-deleted files so the position stays sensible after trashing.
        let total = self.items.len() - self.deleted.len();
        let ordinal = (0..=self.current_index)
            .filter(|i| !self.deleted.contains(i))
            .count();
        let title = format!("[{ordinal}/{total}] {name} - Kagami");
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
        self.last_title_index = Some(self.current_index);
    }
}

impl eframe::App for KagamiApp {
    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root_ui.ctx().clone();

        // Files opened via Finder / "Open With" (Apple Event) while we're already
        // running: switch to that selection and raise the window.
        #[cfg(target_os = "macos")]
        {
            open_docs::ensure_installed();
            let launched_at = *self.first_ui_at.get_or_insert_with(std::time::Instant::now);
            if launched_at.elapsed() < Duration::from_millis(2000) {
                ctx.request_repaint_after(Duration::from_millis(250));
            }
            let opened = open_docs::drain();
            if !opened.is_empty() {
                self.open_paths(opened);
                self.request_focus(500);
            }
        }

        // Pull the app and window to the front. A single request can land
        // before the window is ready or before the closing startup picker has
        // handed activation back, so re-assert until the deadline passes.
        // ViewportCommand::Focus alone does not re-activate a deactivated app
        // on macOS, so activate explicitly as well.
        if let Some(deadline) = self.focus_until {
            if std::time::Instant::now() < deadline {
                activate_app();
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                ctx.request_repaint();
            } else {
                self.focus_until = None;
            }
        }

        // The background folder scan finished: install its browsing list.
        if let Some(rx) = &self.scan_rx
            && let Ok(result) = rx.try_recv()
        {
            self.scan_rx = None;
            self.scanning = false;
            self.install_scanned(result);
        }

        // Fold in any archive/PDF that finished expanding.
        while let Ok(res) = self.expand_rx.try_recv() {
            self.apply_expansion(res);
        }

        while let Ok((index, decoded)) = self.decoded_rx.try_recv() {
            self.cache.entry(index).or_insert_with(|| {
                let size = [decoded.width as usize, decoded.height as usize];
                let mut frames = Vec::with_capacity(decoded.frames.len());
                let mut delays = Vec::with_capacity(decoded.frames.len());
                for (i, (rgba, delay)) in decoded.frames.iter().enumerate() {
                    let img = egui::ColorImage::from_rgba_unmultiplied(size, rgba);
                    frames.push(ctx.load_texture(
                        format!("Kagami_{index}_{i}"),
                        img,
                        egui::TextureOptions::LINEAR,
                    ));
                    delays.push(*delay);
                }
                CachedImage { frames, delays }
            });
        }

        egui::CentralPanel::no_frame()
            .frame(egui::Frame::NONE.fill(egui::Color32::BLACK))
            .show_inside(root_ui, |ui| {
                // N reopens the file/folder picker to browse a new selection;
                // cancelling keeps the current one. Handled before the
                // empty-list return so it also works from "No images found".
                // The modal steals activation, so re-assert focus afterwards
                // on the same time budget used at launch.
                if ui.input(|i| i.key_pressed(egui::Key::N)) {
                    let picked = pick_paths();
                    if !picked.is_empty() {
                        self.open_paths(picked);
                    }
                    self.request_focus(800);
                }

                if self.items.is_empty() || self.deleted.len() == self.items.len() {
                    let message = if self.scanning {
                        "Scanning…"
                    } else {
                        "No images found — press N to open"
                    };
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new(message)
                                .color(egui::Color32::WHITE)
                                .size(24.0),
                        );
                    });
                    return;
                }

                // Cmd+Delete (Cmd+Backspace, the Mac "delete" key) trashes the
                // current file. Checked before navigation so Cmd doesn't also
                // step to the previous file.
                let cmd = ui.input(|i| i.modifiers.command);
                if cmd
                    && ui.input(|i| {
                        i.key_pressed(egui::Key::Backspace) || i.key_pressed(egui::Key::Delete)
                    })
                {
                    self.delete_current();
                }

                // File navigation with arrows/Backspace, skipping trashed files.
                // Space is handled below: "next" for an image, "play/pause" for a
                // video.
                let mut changed = false;
                if ui.input(|i| i.key_pressed(egui::Key::ArrowRight))
                    && let Some(n) = self.neighbor(self.current_index, 1)
                {
                    self.current_index = n;
                    changed = true;
                }
                if !cmd
                    && ui.input(|i| {
                        i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::Backspace)
                    })
                    && let Some(n) = self.neighbor(self.current_index, -1)
                {
                    self.current_index = n;
                    changed = true;
                }

                let space = ui.input(|i| i.key_pressed(egui::Key::Space));
                if space
                    && !is_video_file(self.items[self.current_index].media_name())
                    && let Some(n) = self.neighbor(self.current_index, 1)
                {
                    self.current_index = n;
                    changed = true;
                }

                if changed {
                    self.shared
                        .current_idx
                        .store(self.current_index, Ordering::Relaxed);
                    self.prune_cache();
                    self.reset_zoom();
                }

                // Read any archive/PDF the view has come near (and only those).
                self.maybe_expand();

                self.update_title(ui.ctx());

                // Fullscreen: Enter or `f` toggles, double-click on the video
                // toggles too (below). Esc leaves fullscreen, or quits the app
                // when already windowed.
                let (toggle_fs, esc) = ui.input(|i| {
                    (
                        i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::F),
                        i.key_pressed(egui::Key::Escape),
                    )
                });
                if toggle_fs {
                    self.set_fullscreen(ui.ctx(), !self.fullscreen);
                } else if esc {
                    if self.fullscreen {
                        self.set_fullscreen(ui.ctx(), false);
                    } else {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }

                // Open/close the video player to match the current item.
                self.sync_video(&ctx);

                let rect = ui.max_rect();
                let cur = self.current_index;
                let cur_is_video = is_video_file(self.items[cur].media_name());

                #[cfg(target_os = "macos")]
                if !cur_is_video && let Some(c) = &self.native_controls {
                    c.hide();
                }
                if cur_is_video {
                    let (seek_back, seek_fwd, mute, save, rot_cw, rot_ccw, rot_reset, time, moved) =
                        ui.input(|i| {
                            (
                                i.key_pressed(egui::Key::ArrowUp),
                                i.key_pressed(egui::Key::ArrowDown),
                                i.key_pressed(egui::Key::M),
                                i.key_pressed(egui::Key::S),
                                i.key_pressed(egui::Key::R),
                                i.key_pressed(egui::Key::E),
                                i.key_pressed(egui::Key::T),
                                i.time,
                                i.pointer.delta() != egui::Vec2::ZERO,
                            )
                        });
                    let (sub_smaller, sub_larger, sub_toggle) = ui.input(|i| {
                        (
                            i.key_pressed(egui::Key::Minus),
                            i.key_pressed(egui::Key::Equals) || i.key_pressed(egui::Key::Plus),
                            i.key_pressed(egui::Key::Num0),
                        )
                    });
                    // Reveal the controls on any pointer movement or control key,
                    // then let them fade out after a few idle seconds of playback.
                    let acted = seek_back || seek_fwd || mute || space;
                    if moved || acted {
                        self.controls_until = time + 2.5;
                    }

                    let mut subtitle = ui.input(|i| {
                        i.raw
                            .dropped_files
                            .iter()
                            .filter_map(|f| f.path.clone())
                            .find(|p| is_subtitle_file(p))
                    });
                    if self.video.is_some()
                        && ui.input(|i| i.key_pressed(egui::Key::C) && i.modifiers.is_none())
                    {
                        let picked = pick_subtitle();
                        self.request_focus(800);
                        subtitle = picked.or(subtitle);
                    }

                    let mut dbl_fullscreen = false;
                    // Pull any commands the native panel pushed since last frame.
                    // Apply them before snapshotting state for the panel below:
                    // otherwise a seek click can be followed by one stale
                    // `setDoubleValue`, briefly snapping the playhead backwards.
                    #[cfg(target_os = "macos")]
                    let native_cmds = self.drain_native();
                    #[cfg(target_os = "macos")]
                    if native_cmds
                        .iter()
                        .any(|c| matches!(c, NativeCmd::Fullscreen))
                    {
                        self.set_fullscreen(&ctx, !self.fullscreen);
                    }

                    if let Some((_, player)) = &mut self.video {
                        #[cfg(target_os = "macos")]
                        for c in native_cmds {
                            match c {
                                NativeCmd::Seek(f) => player.seek_to_fraction(f),
                                NativeCmd::Volume(v) => player.set_volume(v),
                                NativeCmd::Play => player.toggle_pause(),
                                NativeCmd::SeekBack => player.seek_by(-video::SEEK_STEP),
                                NativeCmd::SeekForward => player.seek_by(video::SEEK_STEP),
                                NativeCmd::Mute => player.toggle_mute(),
                                NativeCmd::Fullscreen => {}
                                NativeCmd::SaveFrame => {
                                    if let Some(path) =
                                        downloads_png_path(self.items[cur].media_name())
                                    {
                                        match player.save_snapshot(&path) {
                                            Ok(()) => {
                                                eprintln!("saved frame to {}", path.display())
                                            }
                                            Err(e) => eprintln!("frame save failed: {e}"),
                                        }
                                    }
                                }
                                NativeCmd::AudioTrack(track) => player.set_audio_track(track),
                            }
                        }
                        if let Some(path) = &subtitle {
                            player.set_subtitle(path);
                        }
                        if seek_back {
                            player.seek_by(-video::SEEK_STEP);
                        }
                        if seek_fwd {
                            player.seek_by(video::SEEK_STEP);
                        }
                        if mute {
                            player.toggle_mute();
                        }
                        if rot_cw {
                            player.rotate(1);
                        }
                        if rot_ccw {
                            player.rotate(3);
                        }
                        if rot_reset {
                            player.reset_rotation();
                        }
                        if sub_smaller {
                            self.subtitle_scale =
                                video::step_subtitle_scale(self.subtitle_scale, -1);
                        }
                        if sub_larger {
                            self.subtitle_scale =
                                video::step_subtitle_scale(self.subtitle_scale, 1);
                        }
                        if sub_smaller || sub_larger {
                            player.set_subtitle_scale(self.subtitle_scale);
                        }
                        if sub_toggle {
                            player.toggle_subtitles();
                        }
                        // Save exactly the on-screen frame at full source
                        // resolution (libvlc writes the PNG directly).
                        if save && let Some(path) = downloads_png_path(self.items[cur].media_name())
                        {
                            match player.save_snapshot(&path) {
                                Ok(()) => eprintln!("saved frame to {}", path.display()),
                                Err(e) => eprintln!("frame save failed: {e}"),
                            }
                        }

                        // The frame is drawn by libvlc's native NSView over the
                        // window; egui only drives the controls/input below.
                        player.update(&ctx);

                        #[cfg_attr(target_os = "macos", allow(unused_variables))]
                        let show_controls = player.is_paused() || time < self.controls_until;

                        // The native IINA-style floating panel: positioned and
                        // state-pushed here, but input handled by AppKit. Build
                        // the state only after native commands were applied so a
                        // seek action is reflected immediately in the slider.
                        #[cfg(target_os = "macos")]
                        {
                            let position = player.position();
                            let duration = player.duration();
                            let paused = player.is_paused();
                            let muted = player.is_muted();
                            let volume = player.volume();
                            let current_audio_track = player.current_audio_track();
                            let audio_tracks = player.audio_tracks();
                            let panel_state = controls::VideoState {
                                position,
                                duration,
                                paused,
                                muted,
                                volume,
                                audio_tracks,
                                current_audio_track,
                                visible: paused || time < self.controls_until,
                            };
                            if self.native_controls.is_none()
                                && let Some(mtm) = objc2::MainThreadMarker::new()
                                && let Some(host) = &self.video_host
                            {
                                let c = controls::NativeControls::new(mtm, host.view());
                                c.set_callbacks(native_controls_callbacks(&self.shared, &ctx));
                                self.native_controls = Some(c);
                            }
                            let screen = ui.ctx().content_rect();
                            let pos = self.playback_panel_pos.map(|p| (p.horizontal, p.vertical));
                            if let Some(c) = &self.native_controls {
                                c.update(
                                    screen.width() as f64,
                                    screen.height() as f64,
                                    &panel_state,
                                    &fmt_time(panel_state.position),
                                    &fmt_time(panel_state.duration),
                                    pos,
                                );
                            }
                        }

                        // One response covers the whole video for click-to-pause
                        // and double-click fullscreen. The panel itself (sliders,
                        // play button, drag) is fully native AppKit — IINA's OSC —
                        // so egui no longer drives any of it.
                        let resp =
                            ui.interact(rect, ui.id().with("video"), egui::Sense::click_and_drag());
                        #[cfg_attr(target_os = "macos", allow(unused_variables))]
                        let pointer = resp.interact_pointer_pos();

                        #[cfg(not(target_os = "macos"))]
                        if show_controls {
                            let info = VideoOverlay {
                                position: player.position(),
                                duration: player.duration(),
                                paused: player.is_paused(),
                                muted: player.is_muted(),
                                volume: player.volume(),
                            };
                            let scrubbing = resp.dragged()
                                && pointer.is_some_and(|p| p.y >= rect.bottom() - CONTROLS_STRIP_H);
                            draw_video_overlay(ui, rect, &info, scrubbing);
                        }

                        // Double-click toggles fullscreen (IINA-style); a single
                        // click (or Space) toggles play.
                        dbl_fullscreen = resp.double_clicked();
                        if (space && !changed) || (resp.clicked() && !resp.double_clicked()) {
                            player.toggle_pause();
                        }
                    } else {
                        let msg = self
                            .video_error
                            .clone()
                            .unwrap_or_else(|| "Loading video...".to_string());
                        ui.centered_and_justified(|ui| {
                            ui.label(
                                egui::RichText::new(msg)
                                    .color(egui::Color32::WHITE)
                                    .size(20.0),
                            );
                        });
                    }
                    if dbl_fullscreen {
                        self.set_fullscreen(&ctx, !self.fullscreen);
                    }
                } else if let Some(cached) = self.cache.get(&cur) {
                    // Animated GIFs pick the frame for the current clock and
                    // schedule a repaint at the next frame flip; stills return
                    // an infinite delay, so nothing extra is scheduled.
                    let (fi, next_in) = cached.frame_at(ui.input(|i| i.time));
                    if next_in.is_finite() {
                        ctx.request_repaint_after(Duration::from_secs_f32(next_in.max(0.005)));
                    }
                    // Clone the cheap (Arc-backed) handle so `self` is free to be
                    // borrowed mutably for the zoom/pan state during painting.
                    let texture = cached.frames[fi].clone();
                    self.show_image(ui, rect, &texture);
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.add(egui::Spinner::new().size(40.0));
                    });
                    ctx.request_repaint_after(Duration::from_millis(16));
                }
            });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, PANEL_POS_KEY, &self.playback_panel_pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::jpeg::JpegEncoder;

    fn encode_jpeg(w: u32, h: u32) -> Vec<u8> {
        let buf = image::RgbImage::from_pixel(w, h, image::Rgb([120, 80, 40]));
        let mut out = Vec::new();
        JpegEncoder::new(&mut out)
            .encode(&buf, w, h, image::ExtendedColorType::Rgb8)
            .unwrap();
        out
    }

    #[test]
    fn decodes_jpeg_wider_than_old_zune_limit() {
        // 17000 > the old 16384 default guard that made big photos fail forever.
        let data = encode_jpeg(17000, 64);
        let d = decode_from_bytes(&data, 4096).expect("should decode");
        assert!(
            d.width <= 4096 && d.height <= 4096,
            "downscaled: {}x{}",
            d.width,
            d.height
        );
        assert_eq!(d.frames[0].0.len(), (d.width * d.height * 4) as usize);
    }

    #[test]
    fn animated_gif_decodes_every_frame_with_delays() {
        use image::codecs::gif::{GifEncoder, Repeat};
        use image::{Delay, Frame, Rgba, RgbaImage};
        let mut data = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut data);
            enc.set_repeat(Repeat::Infinite).unwrap();
            for c in [0u8, 128, 255] {
                let img = RgbaImage::from_pixel(8, 8, Rgba([c, 0, 0, 255]));
                let delay = Delay::from_numer_denom_ms(200, 1);
                enc.encode_frame(Frame::from_parts(img, 0, 0, delay))
                    .unwrap();
            }
        }
        let d = decode_from_bytes(&data, 4096).expect("gif decodes");
        assert_eq!(d.frames.len(), 3, "all frames kept");
        assert!(
            d.frames
                .iter()
                .all(|(_, delay)| (*delay - 0.2).abs() < 0.02)
        );
        // The animation clock walks frames by their delays and wraps around.
        let clock = CachedImage {
            frames: Vec::new(),
            delays: d.frames.iter().map(|f| f.1).collect(),
        };
        assert_eq!(clock.frame_at(0.0).0, 0);
        assert_eq!(clock.frame_at(0.25).0, 1);
        assert_eq!(clock.frame_at(0.45).0, 2);
        assert_eq!(clock.frame_at(0.65).0, 0, "wraps after the last frame");
        let (_, next_in) = clock.frame_at(0.25);
        assert!((next_in - 0.15).abs() < 0.02, "repaint lands on the flip");
    }

    #[test]
    fn natural_order_sorts_numerically() {
        let mut v = vec![
            "no.126_010.jpg",
            "no.126_002.jpg",
            "no.126_1.jpg",
            "no.126_009.jpg",
            "no.126_100.jpg",
            "no.126_20.jpg",
        ];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(
            v,
            vec![
                "no.126_1.jpg",
                "no.126_002.jpg",
                "no.126_009.jpg",
                "no.126_010.jpg",
                "no.126_20.jpg",
                "no.126_100.jpg",
            ]
        );
    }

    #[test]
    fn archive_entries_list_and_decode_in_memory() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!("kagami_test_{}.cbz", std::process::id()));
        let opts = zip::write::SimpleFileOptions::default();
        let mut zw = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        zw.start_file("pages/page1.jpg", opts).unwrap();
        zw.write_all(&encode_jpeg(64, 32)).unwrap();
        zw.start_file("__MACOSX/._page1.jpg", opts).unwrap();
        zw.write_all(b"resource fork junk").unwrap();
        zw.start_file("notes.txt", opts).unwrap();
        zw.write_all(b"not media").unwrap();
        zw.finish().unwrap();

        let items = scan_archive(&path, is_media_name);
        assert_eq!(items.len(), 1, "junk and non-media entries are skipped");
        assert!(
            items[0]
                .display(Path::new("/"))
                .ends_with(".cbz/pages/page1.jpg")
        );

        let data = items[0]
            .read(&|| false)
            .expect("entry decompresses to memory");
        let d = decode_from_bytes(&data, 4096).expect("entry decodes");
        assert_eq!((d.width, d.height), (64, 32));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn downscale_caps_both_sides() {
        let rgba = vec![255u8; (9000 * 9000 * 4) as usize];
        let (w, h, out) = downscale_to_limit(9000, 9000, rgba, 4096).unwrap();
        assert_eq!((w, h), (4096, 4096));
        assert_eq!(out.len(), (4096 * 4096 * 4) as usize);
    }
}

/// Kagami's icon as raw RGBA, decoded from the same `assets/icon.png` the .app's
/// AppIcon.icns is built from. eframe hands this to macOS's native
/// `NSApplication setApplicationIconImage` at runtime; without it eframe falls
/// back to its generic default icon, which replaces the Dock icon the moment the
/// viewer window opens (so it no longer matches the icns shown by the startup
/// folder picker). We pass decoded RGBA rather than the PNG on purpose: eframe
/// builds the NSImage from raw bytes to avoid a macOS libpng-load crash.
fn load_icon() -> egui::IconData {
    let image = image::load_from_memory(include_bytes!("../assets/icon.png"))
        .expect("bundled icon is a valid PNG")
        .into_rgba8();
    let (width, height) = image.dimensions();
    egui::IconData {
        rgba: image.into_raw(),
        width,
        height,
    }
}

/// Bring the app to the foreground. Used after the launch picker and when a
/// file is opened into a running instance: on macOS the window focus request
/// alone does not re-activate the app if another one holds activation.
#[cfg(target_os = "macos")]
fn activate_app() {
    use objc2::MainThreadMarker;
    use objc2_app_kit::NSApplication;

    if let Some(mtm) = MainThreadMarker::new() {
        #[allow(deprecated)]
        NSApplication::sharedApplication(mtm).activateIgnoringOtherApps(true);
    }
}

#[cfg(not(target_os = "macos"))]
fn activate_app() {}

/// Prompt for files *or* a folder to open, returning an empty Vec if cancelled.
/// On macOS this is a native NSOpenPanel (which, unlike rfd, can accept either)
/// and it also makes the process a regular, foreground app so the panel and the
/// window that follows come to the front.
#[cfg(target_os = "macos")]
fn pick_paths() -> Vec<PathBuf> {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{
        NSApplication, NSApplicationActivationPolicy, NSModalResponseOK, NSOpenPanel,
    };

    // `main` runs on the main thread, which AppKit requires for UI calls.
    let Some(mtm) = MainThreadMarker::new() else {
        return Vec::new();
    };
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    #[allow(deprecated)]
    app.activateIgnoringOtherApps(true);

    let panel = NSOpenPanel::openPanel(mtm);
    panel.setCanChooseFiles(true);
    panel.setCanChooseDirectories(true);
    panel.setAllowsMultipleSelection(true);
    if panel.runModal() != NSModalResponseOK {
        return Vec::new();
    }
    panel
        .URLs()
        .iter()
        .filter_map(|url| url.path().map(|p| PathBuf::from(p.to_string())))
        .collect()
}

#[cfg(target_os = "macos")]
fn launch_selection() -> Vec<PathBuf> {
    let mut paths = open_docs::drain();
    if paths.is_empty() {
        paths = pick_paths();
    }
    if paths.is_empty() {
        paths = open_docs::drain();
    }
    if paths.is_empty() {
        std::process::exit(0);
    }
    paths
}

#[cfg(target_os = "macos")]
fn pick_subtitle() -> Option<PathBuf> {
    use objc2::MainThreadMarker;
    use objc2::rc::Retained;
    use objc2::runtime::{AnyClass, AnyObject};
    use objc2::msg_send;
    use objc2_app_kit::{NSModalResponseOK, NSOpenPanel};
    use objc2_foundation::{NSArray, NSString};

    let mtm = MainThreadMarker::new()?;
    let panel = NSOpenPanel::openPanel(mtm);
    panel.setCanChooseFiles(true);
    panel.setCanChooseDirectories(false);
    panel.setAllowsMultipleSelection(false);

    if let Some(class) = AnyClass::get(c"UTType") {
        let types: Vec<Retained<AnyObject>> = SUBTITLE_EXTENSIONS
            .iter()
            .filter_map(|ext| unsafe {
                msg_send![class, typeWithFilenameExtension: &*NSString::from_str(ext)]
            })
            .collect();
        let array = NSArray::from_retained_slice(&types);
        unsafe {
            let _: () = msg_send![&*panel, setAllowedContentTypes: &*array];
        }
    }

    if panel.runModal() != NSModalResponseOK {
        return None;
    }
    panel
        .URLs()
        .iter()
        .find_map(|url| url.path().map(|p| PathBuf::from(p.to_string())))
}

#[cfg(not(target_os = "macos"))]
fn pick_subtitle() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .set_title("Select subtitle file")
        .add_filter("Subtitles", SUBTITLE_EXTENSIONS)
        .pick_file()
}

/// Folder picker fallback for non-macOS platforms (rfd can't offer a combined
/// file-or-folder dialog).
#[cfg(not(target_os = "macos"))]
fn pick_paths() -> Vec<PathBuf> {
    rfd::FileDialog::new()
        .set_title("Select folders of images")
        .pick_folders()
        .unwrap_or_default()
}

/// macOS file associations: receive the paths Finder hands us when a media file
/// is double-clicked or opened via "Open With Kagami". Finder does not pass these
/// as argv — it sends an `odoc` Apple Event — so we install a handler that
/// collects them and drain it each frame (opens in an already-running instance).
#[cfg(target_os = "macos")]
mod open_docs {
    use std::ffi::c_void;
    use std::path::PathBuf;
    use std::sync::Mutex;

    static OPENED: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
    static REPAINT: std::sync::OnceLock<eframe::egui::Context> = std::sync::OnceLock::new();

    pub fn set_repaint(ctx: eframe::egui::Context) {
        let _ = REPAINT.set(ctx);
    }

    #[repr(C)]
    struct AEDesc {
        descriptor_type: u32,
        data_handle: *mut c_void,
    }
    impl AEDesc {
        fn null() -> Self {
            AEDesc {
                descriptor_type: 0,
                data_handle: std::ptr::null_mut(),
            }
        }
    }

    type AEEventHandler = unsafe extern "C" fn(*const AEDesc, *mut AEDesc, *mut c_void) -> i16;

    /// FourCharCode from a 4-byte literal.
    const fn fourcc(s: &[u8; 4]) -> u32 {
        ((s[0] as u32) << 24) | ((s[1] as u32) << 16) | ((s[2] as u32) << 8) | (s[3] as u32)
    }

    unsafe extern "C" {
        fn AEInstallEventHandler(
            cls: u32,
            id: u32,
            handler: Option<AEEventHandler>,
            refcon: *mut c_void,
            is_sys: u8,
        ) -> i32;
        fn AEGetEventHandler(
            cls: u32,
            id: u32,
            handler: *mut Option<AEEventHandler>,
            refcon: *mut *mut c_void,
            is_sys: u8,
        ) -> i32;
        fn AEGetParamDesc(
            event: *const AEDesc,
            keyword: u32,
            desired: u32,
            result: *mut AEDesc,
        ) -> i32;
        fn AECountItems(list: *const AEDesc, count: *mut isize) -> i32;
        fn AEGetNthDesc(
            list: *const AEDesc,
            index: isize,
            desired: u32,
            keyword: *mut u32,
            result: *mut AEDesc,
        ) -> i32;
        fn AEGetDescDataSize(desc: *const AEDesc) -> isize;
        fn AEGetDescData(desc: *const AEDesc, ptr: *mut c_void, max: isize) -> i32;
        fn AEDisposeDesc(desc: *mut AEDesc) -> i32;
    }

    fn url_to_path(url: &str) -> Option<PathBuf> {
        use objc2_foundation::{NSString, NSURL};
        let s = NSString::from_str(url);
        let nsurl = NSURL::URLWithString(&s)?;
        let p = nsurl.path()?;
        Some(PathBuf::from(p.to_string()))
    }

    unsafe extern "C" fn handle_open(
        event: *const AEDesc,
        _reply: *mut AEDesc,
        _refcon: *mut c_void,
    ) -> i16 {
        unsafe {
            let mut list = AEDesc::null();
            if AEGetParamDesc(event, fourcc(b"----"), fourcc(b"list"), &mut list) != 0 {
                return 0;
            }
            let mut count: isize = 0;
            AECountItems(&list, &mut count);
            for i in 1..=count {
                let mut item = AEDesc::null();
                let mut kw: u32 = 0;
                if AEGetNthDesc(&list, i, fourcc(b"furl"), &mut kw, &mut item) == 0 {
                    let size = AEGetDescDataSize(&item);
                    if size > 0 {
                        let mut buf = vec![0u8; size as usize];
                        if AEGetDescData(&item, buf.as_mut_ptr() as *mut c_void, size) == 0
                            && let Ok(url) = String::from_utf8(buf)
                            && let Some(path) = url_to_path(&url)
                        {
                            OPENED.lock().unwrap().push(path);
                        }
                    }
                    AEDisposeDesc(&mut item);
                }
            }
            AEDisposeDesc(&mut list);
        }
        if let Some(ctx) = REPAINT.get() {
            ctx.request_repaint();
        }
        0
    }

    /// Install the Open-Documents handler.
    pub fn install() {
        unsafe {
            AEInstallEventHandler(
                fourcc(b"aevt"),
                fourcc(b"odoc"),
                Some(handle_open),
                std::ptr::null_mut(),
                0,
            );
        }
    }

    pub fn ensure_installed() {
        let mut current: Option<AEEventHandler> = None;
        let mut refcon: *mut c_void = std::ptr::null_mut();
        let err = unsafe {
            AEGetEventHandler(
                fourcc(b"aevt"),
                fourcc(b"odoc"),
                &mut current,
                &mut refcon,
                0,
            )
        };
        let ours = current.is_some_and(|h| h as usize == handle_open as *const () as usize);
        if err != 0 || !ours {
            install();
        }
    }

    pub fn install_at_launch() {
        use block2::RcBlock;
        use objc2::rc::Retained;
        use objc2::runtime::AnyObject;
        use objc2::{class, msg_send};
        use objc2_app_kit::NSApplicationWillFinishLaunchingNotification;

        let block = RcBlock::new(|_note: std::ptr::NonNull<AnyObject>| install());
        unsafe {
            let center: Retained<AnyObject> =
                msg_send![class!(NSNotificationCenter), defaultCenter];
            let _observer: Option<Retained<AnyObject>> = msg_send![
                &*center,
                addObserverForName: NSApplicationWillFinishLaunchingNotification,
                object: None::<&AnyObject>,
                queue: None::<&AnyObject>,
                usingBlock: &*block
            ];
        }
    }

    /// Files opened since the last call (e.g. "Open With" while already running).
    pub fn drain() -> Vec<PathBuf> {
        std::mem::take(&mut *OPENED.lock().unwrap())
    }
}

fn main() -> Result<()> {
    video::preload();
    #[cfg(target_os = "macos")]
    open_docs::install_at_launch();

    let initial: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    #[cfg(not(target_os = "macos"))]
    let initial = if initial.is_empty() {
        let picked = pick_paths();
        if picked.is_empty() {
            return Ok(());
        }
        picked
    } else {
        initial
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Kagami")
            .with_icon(load_icon())
            .with_maximized(true)
            .with_active(true),
        ..Default::default()
    };

    eframe::run_native(
        "Kagami",
        options,
        Box::new(move |cc| {
            #[cfg(target_os = "macos")]
            let initial = if initial.is_empty() {
                launch_selection()
            } else {
                initial
            };
            Ok(Box::new(KagamiApp::new(cc, initial)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {}", e))?;

    Ok(())
}
