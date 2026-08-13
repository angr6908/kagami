//! Media inside .zip/.cbz archives, browsed in place. Entries are listed from
//! the archive's central directory during the scan, and each image/video is
//! decompressed straight into memory when it is needed — nothing is ever
//! extracted to a temp file.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const ARCHIVE_EXTENSIONS: &[&str] = &["zip", "cbz"];
pub const READ_CHUNK: usize = 1 << 20;

pub fn has_ext(path: &Path, exts: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| exts.iter().any(|c| e.eq_ignore_ascii_case(c)))
        .unwrap_or(false)
}

pub fn is_archive_file(path: &Path) -> bool {
    has_ext(path, ARCHIVE_EXTENSIONS)
}

/// One browsable media item: a plain file on disk, or one entry inside an
/// archive. The whole app browses `Item`s; only the byte-fetching code cares
/// which variant it is.
#[derive(Clone)]
pub enum Item {
    File(PathBuf),
    /// `archive` is shared across all entries of the same archive.
    Archived {
        archive: Arc<PathBuf>,
        entry: String,
    },
    /// One page's embedded image inside an image-only PDF. `doc` (the parsed
    /// PDF) is shared across every page, as `archive` is across zip entries;
    /// `name` is a synthetic image filename giving the page its order and title.
    /// `inner` names the PDF's entry within its archive when the PDF is itself
    /// nested inside a zip/cbz, and is None for a plain PDF file on disk.
    Pdf {
        doc: Arc<crate::pdf::PdfDoc>,
        page: usize,
        inner: Option<String>,
        name: String,
    },
    /// An archive or PDF that has not been opened yet. Reading it to list its
    /// entries forces the whole (often large) file to download on OneDrive, so
    /// it stays a single placeholder slot in the browse list — sorted where the
    /// file lives — until the viewer navigates near it, then it is expanded in
    /// place into its real entries. See `KagamiApp::maybe_expand`.
    Container {
        path: Arc<PathBuf>,
    },
}

impl Item {
    /// The name whose extension decides how the item is handled (image/video).
    pub fn media_name(&self) -> &Path {
        match self {
            Item::File(p) => p,
            Item::Archived { entry, .. } => Path::new(entry),
            Item::Pdf { name, .. } => Path::new(name),
            Item::Container { path } => path,
        }
    }

    /// The file on disk backing this item — the archive itself for archived
    /// entries. This is what hydration touches and what the Trash would take.
    pub fn disk_path(&self) -> &Path {
        match self {
            Item::File(p) => p,
            Item::Archived { archive, .. } => archive,
            Item::Pdf { doc, .. } => doc.disk(),
            Item::Container { path } => path,
        }
    }

    pub fn is_archived(&self) -> bool {
        matches!(self, Item::Archived { .. })
    }

    pub fn is_pdf(&self) -> bool {
        matches!(self, Item::Pdf { .. })
    }

    /// An un-expanded archive/PDF placeholder (see [`Item::Container`]).
    pub fn is_container(&self) -> bool {
        matches!(self, Item::Container { .. })
    }

    /// Title string relative to the opened root: `sub/img.jpg`, or
    /// `sub/comics.cbz/page01.jpg` for an archived entry.
    pub fn display(&self, root: &Path) -> String {
        let disk = self.disk_path();
        let disk = disk.strip_prefix(root).unwrap_or(disk).to_string_lossy();
        match self {
            Item::File(_) => disk.into_owned(),
            Item::Archived { entry, .. } => format!("{disk}/{entry}"),
            // `disk` is the PDF file, or the archive when the PDF is nested; in
            // the nested case `inner` slots the PDF's own name in between.
            Item::Pdf { inner, name, .. } => match inner {
                Some(entry) => format!("{disk}/{entry}/{name}"),
                None => format!("{disk}/{name}"),
            },
            // The archive/PDF file itself while it is still being opened.
            Item::Container { .. } => disk.into_owned(),
        }
    }

    /// Lowercased string the browsing list is natural-sorted by; archive
    /// entries slot in under their archive's own path.
    pub fn sort_key(&self) -> String {
        match self {
            Item::File(p) => p.to_string_lossy().to_ascii_lowercase(),
            Item::Archived { archive, entry } => {
                format!("{}/{entry}", archive.to_string_lossy()).to_ascii_lowercase()
            }
            Item::Pdf {
                doc, inner, name, ..
            } => {
                let disk = doc.disk().to_string_lossy();
                match inner {
                    Some(entry) => format!("{disk}/{entry}/{name}"),
                    None => format!("{disk}/{name}"),
                }
                .to_ascii_lowercase()
            }
            // Sort the placeholder where the archive/PDF file lives, so its
            // entries (keyed `path/entry`) slot in right after it once expanded.
            Item::Container { path } => path.to_string_lossy().to_ascii_lowercase(),
        }
    }

    /// How to hand this item to the video player without a temp file. Stored
    /// (uncompressed) archive entries — the common case for video, which zip
    /// tools rarely re-compress — play straight out of the archive file at a
    /// byte range, costing no RAM; compressed entries have to be decompressed
    /// fully into memory, the only seekable option without extraction.
    pub fn video_source(&self) -> Option<VideoSource> {
        match self {
            Item::File(p) => Some(VideoSource::Path(p.clone())),
            Item::Archived { archive, entry } => {
                let file = std::fs::File::open(archive.as_path()).ok()?;
                let mut zip = zip::ZipArchive::new(BufReader::new(file)).ok()?;
                let entry = zip.by_name(entry).ok()?;
                if entry.compression() == zip::CompressionMethod::Stored
                    && !entry.encrypted()
                    && let Some(start) = entry.data_start()
                {
                    return Some(VideoSource::FileRange {
                        path: archive.as_path().to_path_buf(),
                        start,
                        len: entry.size(),
                    });
                }
                let size = entry.size() as usize;
                read_chunked(entry, size, &|| false).map(VideoSource::Bytes)
            }
            // PDF entries are always images, never video.
            Item::Pdf { .. } => None,
            // Not playable until expanded into its entries.
            Item::Container { .. } => None,
        }
    }

    /// The item's full bytes, decompressing an archived entry into memory.
    /// `cancelled` is polled between chunks so stale background reads bail
    /// out; None means either cancellation or an I/O/archive error.
    pub fn read(&self, cancelled: &dyn Fn() -> bool) -> Option<Vec<u8>> {
        match self {
            Item::File(p) => read_chunked(std::fs::File::open(p).ok()?, 0, cancelled),
            Item::Archived { archive, entry } => {
                let file = std::fs::File::open(archive.as_path()).ok()?;
                let mut zip = zip::ZipArchive::new(BufReader::new(file)).ok()?;
                let entry = zip.by_name(entry).ok()?;
                let size = entry.size() as usize;
                read_chunked(entry, size, cancelled)
            }
            // PDF pages don't use the file-bytes path; they are decoded from
            // the resident document via `pdf_image` in the decode worker.
            Item::Pdf { .. } => None,
            // A placeholder has no displayable bytes; it is expanded, not read.
            Item::Container { .. } => None,
        }
    }

    /// A PDF page's extracted image, decoded straight from the resident
    /// document. None for every non-PDF item (which use [`Item::read`]).
    pub fn pdf_image(&self) -> Option<crate::pdf::PdfImage> {
        match self {
            Item::Pdf { doc, page, .. } => doc.page_image(*page),
            _ => None,
        }
    }
}

/// See [`Item::video_source`].
pub enum VideoSource {
    Path(PathBuf),
    FileRange { path: PathBuf, start: u64, len: u64 },
    Bytes(Vec<u8>),
}

/// List an archive's media entries (per `is_media`), skipping directories and
/// metadata junk. Nested PDFs are expanded in place into their page images, the
/// same way a PDF on disk is. Unreadable/encrypted archives yield an empty list.
pub fn scan_archive(path: &Path, is_media: impl Fn(&Path) -> bool) -> Vec<Item> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let Ok(mut zip) = zip::ZipArchive::new(BufReader::new(file)) else {
        return Vec::new();
    };
    let archive = Arc::new(path.to_path_buf());
    // Snapshot the entry names first so the archive can be re-borrowed mutably
    // below to decompress any nested PDFs.
    let names: Vec<String> = zip
        .file_names()
        .filter(|name| !name.ends_with('/') && !is_junk(name))
        .map(str::to_string)
        .collect();
    let mut items = Vec::new();
    for name in names {
        let p = Path::new(&name);
        if is_media(p) {
            items.push(Item::Archived {
                archive: archive.clone(),
                entry: name,
            });
        } else if crate::pdf::is_pdf_file(p)
            && let Some(bytes) = read_named(&mut zip, &name)
        {
            items.extend(crate::pdf::scan_pdf_bytes(path, &name, &bytes));
        }
    }
    items
}

/// Decompress one archive entry, by name, fully into memory.
fn read_named(zip: &mut zip::ZipArchive<BufReader<std::fs::File>>, name: &str) -> Option<Vec<u8>> {
    let entry = zip.by_name(name).ok()?;
    let size = entry.size() as usize;
    read_chunked(entry, size, &|| false)
}

/// macOS zips are littered with `__MACOSX/` resource forks and `._*`/`.DS_Store`
/// dotfiles; none of them are viewable media even when the extension says so.
fn is_junk(name: &str) -> bool {
    name.split('/')
        .any(|part| part.starts_with('.') || part == "__MACOSX")
}

fn read_chunked(
    mut r: impl Read,
    size_hint: usize,
    cancelled: &dyn Fn() -> bool,
) -> Option<Vec<u8>> {
    let mut data = Vec::with_capacity(size_hint);
    let mut buf = vec![0u8; READ_CHUNK];
    loop {
        if cancelled() {
            return None;
        }
        match r.read(&mut buf) {
            Ok(0) => return Some(data),
            Ok(n) => data.extend_from_slice(&buf[..n]),
            Err(_) => return None,
        }
    }
}
