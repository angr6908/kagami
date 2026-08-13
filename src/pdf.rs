//! Image-only PDFs, browsed in place the same way .zip/.cbz are. A PDF here is
//! treated as nothing more than a bag of images: each page's embedded image
//! XObject is pulled straight out of the file and handed to the normal image
//! decode path. We never rasterise a page — vector text, forms and annotations
//! are ignored — so a PDF that isn't just scanned/exported images yields no
//! viewable entries. The whole document is parsed once at scan time and shared
//! (behind an `Arc`) across every page, mirroring how one `zip` archive backs
//! all of its entries.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lopdf::{Dictionary, Document, Object, ObjectId, Stream};

use crate::archive::{Item, has_ext};

const PDF_EXTENSIONS: &[&str] = &["pdf"];

pub fn is_pdf_file(path: &Path) -> bool {
    has_ext(path, PDF_EXTENSIONS)
}

/// One page's image, extracted in whichever form is cheapest to hand to the
/// decode pipeline: raw images are expanded straight to display-ready RGBA (no
/// re-encode round trip), while an embedded JPEG/JPEG2000 is passed through
/// as-is for the shared image decoder to handle.
pub enum PdfImage {
    Rgba {
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
    Encoded(Vec<u8>),
}

/// A parsed PDF plus, in page order, the object id of each page's main image.
/// Shared across all of the PDF's [`Item`]s; the parsed document stays resident
/// so extracting a page is a single stream decompress, not a re-parse. A PDF
/// nested inside a zip is parsed from the decompressed entry bytes but still
/// points `disk` at the archive, so hydration and the trash guard treat the
/// archive as the backing file exactly as for the archive's own entries.
pub struct PdfDoc {
    disk: PathBuf,
    doc: Document,
    /// One image XObject id per browsable page, in reading order.
    images: Vec<ObjectId>,
}

impl PdfDoc {
    /// The on-disk file backing this PDF: the PDF itself, or the archive it
    /// lives inside.
    pub fn disk(&self) -> &Path {
        &self.disk
    }

    pub fn page_count(&self) -> usize {
        self.images.len()
    }

    /// The extracted image for `page`: display-ready RGBA for a raw image, or
    /// the embedded JPEG/JPEG2000 bytes for a DCT/JPX one. None for a page whose
    /// image uses an encoding we can't turn into pixels.
    pub fn page_image(&self, page: usize) -> Option<PdfImage> {
        let id = *self.images.get(page)?;
        let stream = self.doc.get_object(id).ok()?.as_stream().ok()?;
        extract_image(&self.doc, stream)
    }
}

/// List a PDF file's per-page images as browsable [`Item`]s. A PDF with no
/// extractable image (born-digital text, or encodings we don't support)
/// yields an empty list, exactly like an unreadable archive.
pub fn scan_pdf(path: &Path) -> Vec<Item> {
    match Document::load(path) {
        Ok(doc) => build_items(doc, path.to_path_buf(), None),
        Err(_) => Vec::new(),
    }
}

/// Same as [`scan_pdf`], for a PDF that lives inside an archive: parsed from the
/// decompressed entry `bytes`, backed on disk by the archive at `disk`, and
/// titled under the archive as `disk/entry/<page>`.
pub fn scan_pdf_bytes(disk: &Path, entry: &str, bytes: &[u8]) -> Vec<Item> {
    match Document::load_mem(bytes) {
        Ok(doc) => build_items(doc, disk.to_path_buf(), Some(entry.to_string())),
        Err(_) => Vec::new(),
    }
}

fn build_items(doc: Document, disk: PathBuf, inner: Option<String>) -> Vec<Item> {
    let mut images = Vec::new();
    // get_pages() is ordered by page number (a BTreeMap), so pages come out in
    // reading order and the browsing list needs no extra sort.
    for (_, page_id) in doc.get_pages() {
        if let Some(id) = page_image(&doc, page_id) {
            images.push(id);
        }
    }
    if images.is_empty() {
        return Vec::new();
    }
    let width = decimal_width(images.len());
    let doc = Arc::new(PdfDoc { disk, doc, images });
    (0..doc.page_count())
        .map(|page| Item::Pdf {
            doc: doc.clone(),
            page,
            inner: inner.clone(),
            // A synthetic name so the item reads as an image (extension gates
            // decode scheduling) and sorts/titles in natural page order.
            name: format!("{:0width$}.bmp", page + 1),
        })
        .collect()
}

fn decimal_width(n: usize) -> usize {
    n.to_string().len()
}

/// The object id of a page's largest image XObject, if any. Only the page's own
/// (possibly inherited) Resources are consulted; images nested inside Form
/// XObjects aren't chased, which is fine for the scanned-page PDFs this targets.
fn page_image(doc: &Document, page_id: ObjectId) -> Option<ObjectId> {
    let xobjects = page_xobjects(doc, page_id)?;
    let mut best: Option<(i64, ObjectId)> = None;
    for (_, value) in xobjects.iter() {
        let Ok(id) = value.as_reference() else {
            continue;
        };
        let Ok(stream) = doc.get_object(id).and_then(Object::as_stream) else {
            continue;
        };
        if !is_image(doc, &stream.dict) {
            continue;
        }
        let area = dict_int(doc, &stream.dict, b"Width").unwrap_or(0)
            * dict_int(doc, &stream.dict, b"Height").unwrap_or(0);
        if best.map(|(a, _)| area > a).unwrap_or(true) {
            best = Some((area, id));
        }
    }
    best.map(|(_, id)| id)
}

/// A page's XObject dictionary, resolving Resources inherited from an ancestor
/// in the page tree (the spec allows /Resources on any parent node).
fn page_xobjects(doc: &Document, page_id: ObjectId) -> Option<&Dictionary> {
    let mut node = page_id;
    for _ in 0..64 {
        let dict = doc.get_dictionary(node).ok()?;
        if let Some(res) = dict
            .get(b"Resources")
            .ok()
            .and_then(|o| resolve_dict(doc, o))
            && let Some(xo) = res.get(b"XObject").ok().and_then(|o| resolve_dict(doc, o))
        {
            return Some(xo);
        }
        // Walk up to the parent page-tree node looking for inherited Resources.
        match dict.get(b"Parent").and_then(Object::as_reference) {
            Ok(parent) => node = parent,
            Err(_) => break,
        }
    }
    None
}

fn is_image(doc: &Document, dict: &Dictionary) -> bool {
    dict.get(b"Subtype")
        .ok()
        .and_then(|o| resolve(doc, o))
        .and_then(|o| o.as_name().ok())
        .map(|n| n == b"Image")
        .unwrap_or(false)
}

/// Extract one image XObject. DCT/JPX images are already whole JPEG/JPEG2000
/// streams, so their bytes pass through untouched for the shared decoder;
/// everything else is inflated and expanded directly to RGBA here.
fn extract_image(doc: &Document, stream: &Stream) -> Option<PdfImage> {
    let dict = &stream.dict;
    let filters = filter_names(doc, dict);
    // The last filter in the chain is the image's own pixel codec.
    match filters.last().map(Vec::as_slice) {
        Some(b"DCTDecode") | Some(b"JPXDecode") => {
            return Some(PdfImage::Encoded(stream.content.clone()));
        }
        _ => {}
    }

    let width = dict_int(doc, dict, b"Width")? as u32;
    let height = dict_int(doc, dict, b"Height")? as u32;
    let bpc = dict_int(doc, dict, b"BitsPerComponent").unwrap_or(8);
    if width == 0 || height == 0 {
        return None;
    }
    // Inflate Flate/LZW/etc.; unsupported codecs (CCITT, JBIG2) error out here
    // and the page is simply dropped.
    let raw = stream.decompressed_content().ok()?;
    let cs = color_space(doc, dict.get(b"ColorSpace").ok()?)?;
    let data = samples_to_rgba(&raw, width, height, bpc, &cs)?;
    Some(PdfImage::Rgba {
        width,
        height,
        data,
    })
}

/// A PDF colour space, reduced to what we need to expand samples to RGB.
enum ColorSpace {
    Gray,
    Rgb,
    Cmyk,
    /// A palette: `base` component count and the packed lookup table.
    Indexed {
        base: usize,
        palette: Vec<u8>,
    },
}

fn color_space(doc: &Document, obj: &Object) -> Option<ColorSpace> {
    match resolve(doc, obj)? {
        Object::Name(name) => named_space(name),
        Object::Array(arr) => {
            let head = arr.first().and_then(|o| o.as_name().ok())?;
            match head {
                b"ICCBased" => {
                    // The ICC stream's /N gives the component count directly.
                    let stream = arr.get(1).and_then(|o| resolve(doc, o))?.as_stream().ok()?;
                    match dict_int(doc, &stream.dict, b"N")? {
                        1 => Some(ColorSpace::Gray),
                        4 => Some(ColorSpace::Cmyk),
                        _ => Some(ColorSpace::Rgb),
                    }
                }
                b"CalRGB" | b"Lab" => Some(ColorSpace::Rgb),
                b"CalGray" => Some(ColorSpace::Gray),
                b"Indexed" | b"I" => {
                    let base = arr.get(1).and_then(|o| color_space(doc, o))?;
                    let base = components(&base);
                    let palette = palette_bytes(doc, arr.get(3)?)?;
                    Some(ColorSpace::Indexed { base, palette })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn named_space(name: &[u8]) -> Option<ColorSpace> {
    match name {
        b"DeviceGray" | b"G" | b"CalGray" => Some(ColorSpace::Gray),
        b"DeviceRGB" | b"RGB" | b"CalRGB" => Some(ColorSpace::Rgb),
        b"DeviceCMYK" | b"CMYK" => Some(ColorSpace::Cmyk),
        _ => None,
    }
}

fn components(cs: &ColorSpace) -> usize {
    match cs {
        ColorSpace::Gray | ColorSpace::Indexed { .. } => 1,
        ColorSpace::Rgb => 3,
        ColorSpace::Cmyk => 4,
    }
}

/// The indexed-palette lookup table, which the spec allows as either a byte
/// string or a stream.
fn palette_bytes(doc: &Document, obj: &Object) -> Option<Vec<u8>> {
    match resolve(doc, obj)? {
        Object::String(bytes, _) => Some(bytes.clone()),
        Object::Stream(s) => s.decompressed_content().ok(),
        _ => None,
    }
}

/// Expand raw image samples to packed RGBA8 (opaque alpha), the exact layout
/// the decode pipeline uploads as a texture. Only the sample layouts that
/// scanned/exported PDFs actually use are handled (8-bit for every colour
/// space, plus 1-bit greyscale); anything else returns None and drops the page.
fn samples_to_rgba(
    raw: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    cs: &ColorSpace,
) -> Option<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    let px = w.checked_mul(h)?;
    // Pre-fill with opaque alpha; the colour loops only touch the RGB bytes.
    let mut out = vec![255u8; px.checked_mul(4)?];

    match (bpc, cs) {
        (8, ColorSpace::Gray) => {
            if raw.len() < px {
                return None;
            }
            for i in 0..px {
                let g = raw[i];
                out[i * 4] = g;
                out[i * 4 + 1] = g;
                out[i * 4 + 2] = g;
            }
        }
        (8, ColorSpace::Rgb) => {
            if raw.len() < px * 3 {
                return None;
            }
            for i in 0..px {
                out[i * 4..i * 4 + 3].copy_from_slice(&raw[i * 3..i * 3 + 3]);
            }
        }
        (8, ColorSpace::Cmyk) => {
            if raw.len() < px * 4 {
                return None;
            }
            for i in 0..px {
                let (c, m, y, k) = (raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]);
                out[i * 4] = cmyk_to_rgb(c, k);
                out[i * 4 + 1] = cmyk_to_rgb(m, k);
                out[i * 4 + 2] = cmyk_to_rgb(y, k);
            }
        }
        (8, ColorSpace::Indexed { base, palette }) => {
            if raw.len() < px {
                return None;
            }
            for i in 0..px {
                write_palette(&mut out[i * 4..], raw[i] as usize, *base, palette);
            }
        }
        (1, ColorSpace::Gray) => {
            // 1-bit rows are padded to a byte boundary; 0 is black, 1 is white.
            let stride = w.div_ceil(8);
            if raw.len() < stride * h {
                return None;
            }
            for y in 0..h {
                for x in 0..w {
                    let bit = (raw[y * stride + x / 8] >> (7 - (x % 8))) & 1;
                    let v = if bit == 1 { 255 } else { 0 };
                    let o = (y * w + x) * 4;
                    out[o] = v;
                    out[o + 1] = v;
                    out[o + 2] = v;
                }
            }
        }
        _ => return None,
    }
    Some(out)
}

/// Naive multiplicative CMYK→RGB: enough for the greyscale-plus-tint scans this
/// path sees, without pulling in colour management.
fn cmyk_to_rgb(c: u8, k: u8) -> u8 {
    ((255 - c as u16) * (255 - k as u16) / 255) as u8
}

fn write_palette(out: &mut [u8], index: usize, base: usize, palette: &[u8]) {
    let at = index * base;
    match base {
        1 => {
            let g = palette.get(at).copied().unwrap_or(0);
            out[0] = g;
            out[1] = g;
            out[2] = g;
        }
        4 => {
            let c = palette.get(at).copied().unwrap_or(0);
            let m = palette.get(at + 1).copied().unwrap_or(0);
            let y = palette.get(at + 2).copied().unwrap_or(0);
            let k = palette.get(at + 3).copied().unwrap_or(0);
            out[0] = cmyk_to_rgb(c, k);
            out[1] = cmyk_to_rgb(m, k);
            out[2] = cmyk_to_rgb(y, k);
        }
        _ => {
            out[0] = palette.get(at).copied().unwrap_or(0);
            out[1] = palette.get(at + 1).copied().unwrap_or(0);
            out[2] = palette.get(at + 2).copied().unwrap_or(0);
        }
    }
}

fn filter_names(doc: &Document, dict: &Dictionary) -> Vec<Vec<u8>> {
    let Ok(obj) = dict.get(b"Filter") else {
        return Vec::new();
    };
    match resolve(doc, obj) {
        Some(Object::Name(n)) => vec![n.clone()],
        Some(Object::Array(arr)) => arr
            .iter()
            .filter_map(|o| {
                resolve(doc, o)
                    .and_then(|o| o.as_name().ok())
                    .map(<[u8]>::to_vec)
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn dict_int(doc: &Document, dict: &Dictionary, key: &[u8]) -> Option<i64> {
    resolve(doc, dict.get(key).ok()?)?.as_i64().ok()
}

fn resolve_dict<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Dictionary> {
    resolve(doc, obj)?.as_dict().ok()
}

/// Follow one indirect reference to the object it names; pass other objects
/// through unchanged.
fn resolve<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Object> {
    match obj {
        Object::Reference(id) => doc.get_object(*id).ok(),
        other => Some(other),
    }
}
