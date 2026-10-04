use std::path::{Path, PathBuf};

const DEFAULT_FONTSIZE_FIELD: usize = 2;
const DEFAULT_EVENT_FIELDS: usize = 10;

#[derive(PartialEq)]
enum Section {
    Styles,
    Events,
    Other,
}

pub fn is_ass_file(path: &Path) -> bool {
    crate::archive::has_ext(path, &["ass", "ssa"])
}

pub fn scale_font_sizes(script: &str, percent: i64) -> String {
    let factor = percent as f64 / 100.0;
    let mut out = String::with_capacity(script.len() + script.len() / 16);
    let mut section = Section::Other;
    let mut fontsize_field = DEFAULT_FONTSIZE_FIELD;
    let mut event_fields = DEFAULT_EVENT_FIELDS;

    for line in script.split_inclusive('\n') {
        let body = line.trim_end_matches(['\r', '\n']);
        let ending = &line[body.len()..];
        let trimmed = body.trim_start_matches('\u{feff}').trim_start();

        if trimmed.starts_with('[') {
            let name = trimmed.to_ascii_lowercase();
            section = if name.starts_with("[v4+ styles]") || name.starts_with("[v4 styles]") {
                Section::Styles
            } else if name.starts_with("[events]") {
                Section::Events
            } else {
                Section::Other
            };
            out.push_str(line);
            continue;
        }

        match section {
            Section::Styles => {
                if let Some(rest) = strip_key(trimmed, "Format:") {
                    if let Some(i) = rest
                        .split(',')
                        .position(|f| f.trim().eq_ignore_ascii_case("fontsize"))
                    {
                        fontsize_field = i;
                    }
                } else if let Some(rest) = strip_key(trimmed, "Style:") {
                    let prefix = &body[..body.len() - rest.len()];
                    out.push_str(prefix);
                    out.push_str(&scale_style_fields(rest, fontsize_field, factor));
                    out.push_str(ending);
                    continue;
                }
            }
            Section::Events => {
                if let Some(rest) = strip_key(trimmed, "Format:") {
                    event_fields = rest.split(',').count().max(1);
                } else if let Some(rest) = strip_key(trimmed, "Dialogue:") {
                    let prefix = &body[..body.len() - rest.len()];
                    out.push_str(prefix);
                    out.push_str(&scale_dialogue(rest, event_fields, factor));
                    out.push_str(ending);
                    continue;
                }
            }
            Section::Other => {}
        }
        out.push_str(line);
    }
    out
}

pub fn write_scaled_copy(path: &Path, percent: i64) -> Option<PathBuf> {
    use std::hash::{Hash, Hasher};

    let script = std::fs::read_to_string(path).ok()?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    let dir = std::env::temp_dir().join("Kagami");
    std::fs::create_dir_all(&dir).ok()?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("ass");
    let out = dir.join(format!("sub-{:016x}-{percent}.{ext}", hasher.finish()));
    std::fs::write(&out, scale_font_sizes(&script, percent)).ok()?;
    Some(out)
}

fn strip_key<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let head = line.get(..key.len())?;
    head.eq_ignore_ascii_case(key).then(|| &line[key.len()..])
}

fn scale_style_fields(fields: &str, index: usize, factor: f64) -> String {
    fields
        .split(',')
        .enumerate()
        .map(|(i, field)| {
            if i != index {
                return field.to_string();
            }
            let value = field.trim();
            match value.parse::<f64>() {
                Ok(size) => {
                    let lead = &field[..field.len() - field.trim_start().len()];
                    format!("{lead}{}", format_size(size * factor))
                }
                Err(_) => field.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn scale_dialogue(fields: &str, count: usize, factor: f64) -> String {
    let mut parts: Vec<&str> = fields.splitn(count, ',').collect();
    if parts.len() < count {
        return fields.to_string();
    }
    let text = scale_override_tags(parts[count - 1], factor);
    parts.pop();
    let mut out = parts.join(",");
    out.push(',');
    out.push_str(&text);
    out
}

fn scale_override_tags(text: &str, factor: f64) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_block = false;
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if c == '{' {
            in_block = true;
        } else if c == '}' {
            in_block = false;
        } else if in_block && rest.starts_with("\\fs") {
            let after = &rest[3..];
            let digits = after
                .find(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
                .unwrap_or(after.len());
            if digits > 0
                && after.as_bytes()[0].is_ascii_digit()
                && let Ok(size) = after[..digits].parse::<f64>()
            {
                out.push_str("\\fs");
                out.push_str(&format_size(size * factor));
                rest = &after[digits..];
                continue;
            }
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

fn format_size(size: f64) -> String {
    let rounded = (size * 100.0).round() / 100.0;
    format!("{}", rounded.max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCRIPT: &str = "\u{feff}[Script Info]\r\nPlayResY: 720\r\n\r\n[V4+ Styles]\r\nFormat: Name, Fontname, Fontsize, PrimaryColour\r\nStyle: Default,微软雅黑,54,&H00FFFFFF\r\nStyle: Danmaku,Source Han Sans JP,64,&H00FFFFFF\r\n\r\n[Events]\r\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\r\nDialogue: 2,00:00:13.35,00:00:21.35,Danmaku,,0,0,0,,{\\an7\\move(1280,64,-280,64)\\fs40\\fscx120\\fsp2}うひょー, \\fs99\r\n";

    #[test]
    fn scales_style_fontsize_and_fs_tags_only() {
        let scaled = scale_font_sizes(SCRIPT, 150);
        assert!(scaled.contains("Style: Default,微软雅黑,81,&H00FFFFFF\r\n"));
        assert!(scaled.contains("Style: Danmaku,Source Han Sans JP,96,&H00FFFFFF\r\n"));
        assert!(
            scaled.contains(
                "{\\an7\\move(1280,64,-280,64)\\fs60\\fscx120\\fsp2}うひょー, \\fs99\r\n"
            )
        );
        assert!(scaled.contains("PlayResY: 720\r\n"));
        assert!(scaled.starts_with('\u{feff}'));
    }

    #[test]
    fn unchanged_at_full_scale() {
        assert_eq!(scale_font_sizes(SCRIPT, 100), SCRIPT);
    }

    #[test]
    fn fractional_sizes_round_to_hundredths() {
        let scaled = scale_font_sizes(SCRIPT, 110);
        assert!(scaled.contains(",59.4,"));
        assert!(scaled.contains(",70.4,"));
        assert!(scaled.contains("\\fs44\\fscx120"));
    }
}
