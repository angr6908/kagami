//! A native AppKit floating video control panel, matching IINA's
//! `OSCFloatingView`. It is a translucent rounded bar that the user drags by
//! its backdrop, with native sliders/buttons inside. Mouse handling follows
//! IINA's `mouseDown`/`mouseDragged`/`mouseUp`, controls fire target/action
//! callbacks main.rs uses to drive libvlc, and the bar's position is persisted
//! across launches by main.rs.

use std::cell::{Cell, RefCell};

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Bool};
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{
    NSAppearance, NSApplication, NSBezelStyle, NSBezierPath, NSButton, NSButtonType, NSColor,
    NSControlSize, NSControlStateValueOff, NSControlStateValueOn, NSEvent, NSFont,
    NSFontWeightMedium, NSFontWeightRegular, NSGlassEffectView, NSGraphicsContext,
    NSHapticFeedbackManager, NSHapticFeedbackPattern, NSHapticFeedbackPerformanceTime,
    NSHapticFeedbackPerformer, NSImage, NSImageScaling, NSImageSymbolConfiguration, NSImageView,
    NSMenu, NSMenuItem, NSShadow, NSSlider, NSSliderCell, NSTextAlignment, NSTextField, NSView,
    NSVisualEffectBlendingMode, NSVisualEffectMaterial, NSVisualEffectState, NSVisualEffectView,
};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};

// ---------------------------------------------------------------------------
// IINA's OSCFloatingView drag logic
// ---------------------------------------------------------------------------

/// Closures the native panel calls back into when a control fires, so main.rs
/// can drive the player. Installed via [`NativeControls::set_callbacks`] and
/// rebound when the app replaces its shared playback state; stored in a
/// thread-local because the view methods run on the main thread
/// (`MainThreadOnly`).
pub struct ControlsCallbacks {
    /// Seek to `fraction` of the duration (0..1). IINA's slider `valueChanged`.
    pub on_seek: Box<dyn Fn(f64)>,
    /// Set volume to `fraction` (0..1).
    pub on_volume: Box<dyn Fn(f32)>,
    /// Toggle play/pause (the play button).
    pub on_play: Box<dyn Fn()>,
    /// Jump back by IINA's arrow-button seek step.
    pub on_seek_back: Box<dyn Fn()>,
    /// Jump forward by IINA's arrow-button seek step.
    pub on_seek_forward: Box<dyn Fn()>,
    /// Toggle mute (the speaker/volume button).
    pub on_mute: Box<dyn Fn()>,
    /// Toggle fullscreen (IINA's full-screen toolbar button).
    pub on_fullscreen: Box<dyn Fn()>,
    /// Save the current video frame.
    pub on_save_frame: Box<dyn Fn()>,
    /// Select the libvlc audio stream with this track id.
    pub on_audio_track: Box<dyn Fn(i32)>,
    /// The bar moved. Values match IINA's persisted OSC preferences: horizontal
    /// centre / video width and bottom edge / video height.
    pub on_move: Box<dyn Fn(f64, f64)>,
}

/// Per-app state for the floating bar, kept in a thread-local (views are
/// `MainThreadOnly`).
struct ControlsState {
    /// Where in the view (view coords) the grab landed; `None` when not
    /// dragging. IINA's `mousePosRelatedToView`.
    mouse_pos: Option<NSPoint>,
    /// Whether the alignment haptic already fired for the current drag.
    align_sent: bool,
    /// The current control callbacks; replaced when shared playback state changes.
    callbacks: Option<ControlsCallbacks>,
    /// Current audio streams and selection, refreshed by `NativeControls::update`.
    audio_tracks: Vec<crate::video::AudioTrack>,
    current_audio_track: i32,
}

thread_local! {
    static CONTROLS_STATE: RefCell<ControlsState> = const {
        RefCell::new(ControlsState {
            mouse_pos: None,
            align_sent: false,
            callbacks: None,
            audio_tracks: Vec::new(),
            current_audio_track: -1,
        })
    };
}

// A full-window host for the floating bar. The host itself is transparent to
// hit testing: AppKit controls inside the bar receive events, while a hit that
// resolves to the host falls through to egui below. This mirrors the
// mouse-transparent native video view in video.rs without making the OSC
// controls themselves transparent.
define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "KagamiControlsOverlay"]
    struct ControlsOverlay;

    impl ControlsOverlay {
        #[unsafe(method(hitTest:))]
        fn hit_test(&self, point: NSPoint) -> *mut NSView {
            let hit: *mut NSView = unsafe { msg_send![super(self), hitTest: point] };
            let this = self as *const ControlsOverlay as *mut NSView;
            if hit == this {
                core::ptr::null_mut()
            } else {
                hit
            }
        }
    }
);

// IINA's VolumeButton is deliberately an NSView containing an NSImageView,
// rather than an NSButton. That keeps the speaker glyph free of button hover
// and pressed-state decoration while still making the full 24x24 area clickable.
define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "KagamiVolumeButton"]
    struct VolumeButton;

    impl VolumeButton {
        #[unsafe(method(mouseDown:))]
        fn mouse_down(&self, _event: &NSEvent) {
            CONTROLS_STATE.with(|s| {
                if let Some(cb) = &s.borrow().callbacks {
                    (cb.on_mute)();
                }
            });
        }
    }
);

// IINA's TranslucentView owns the actual OSC content: on Tahoe the content sits
// inside NSGlassEffectView.contentView, while older macOS places it inside an
// NSVisualEffectView. These subclasses keep that hierarchy but make only the
// backdrop itself transparent to hit testing. Child controls still receive
// native AppKit events; empty glass/frosted areas fall through to FloatingBar so
// the user can drag the OSC exactly as in IINA.
define_class!(
    #[unsafe(super(NSVisualEffectView))]
    #[thread_kind = MainThreadOnly]
    #[name = "KagamiVisualEffectBackdrop"]
    struct VisualEffectBackdrop;

    impl VisualEffectBackdrop {
        #[unsafe(method(hitTest:))]
        fn hit_test(&self, point: NSPoint) -> *mut NSView {
            let hit: *mut NSView = unsafe { msg_send![super(self), hitTest: point] };
            let this = self as *const VisualEffectBackdrop as *mut NSView;
            if hit == this {
                core::ptr::null_mut()
            } else {
                hit
            }
        }
    }
);

define_class!(
    #[unsafe(super(NSGlassEffectView))]
    #[thread_kind = MainThreadOnly]
    #[name = "KagamiGlassBackdrop"]
    struct GlassBackdrop;

    impl GlassBackdrop {
        #[unsafe(method(hitTest:))]
        fn hit_test(&self, point: NSPoint) -> *mut NSView {
            let hit: *mut NSView = unsafe { msg_send![super(self), hitTest: point] };
            let this = self as *const GlassBackdrop as *mut NSView;
            if hit == this {
                core::ptr::null_mut()
            } else {
                hit
            }
        }
    }
);

// The floating panel. It IS the bar: its frame is the bar's rect, its subviews
// are the sliders/buttons/labels, and `mouseDown`/`mouseDragged`/`mouseUp`
// move it exactly like IINA's `OSCFloatingView`.
define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "KagamiFloatingBar"]
    struct FloatingBar;

    impl FloatingBar {
        // IINA's `mouseDown`: record where in the view the grab landed, in view
        // coordinates (`mousePosRelatedToView`).
        #[unsafe(method(mouseDown:))]
        fn mouse_down(&self, event: &NSEvent) {
            CONTROLS_STATE.with(|s| {
                let mut s = s.borrow_mut();
                let window = self.window();
                let Some(window) = window.as_ref() else { return };
                // IINA records the *screen* mouse location and subtracts the
                // window's frame origin from it. `NSEvent.mouseLocation` is the
                // bottom-left screen coordinate used by AppKit here.
                let screen_point = NSEvent::mouseLocation();
                let window_frame = window.frame();
                let p = NSPoint::new(
                    screen_point.x - window_frame.origin.x,
                    screen_point.y - window_frame.origin.y,
                );
                let frame_origin = self.frame().origin;
                s.mouse_pos = Some(NSPoint::new(p.x - frame_origin.x, p.y - frame_origin.y));
                // IINA suppresses the alignment tick when a drag begins while
                // the OSC is already within the 5pt centre snap region.
                let content_width = window
                    .contentView()
                    .map(|content| content.bounds().size.width)
                    .unwrap_or(window.frame().size.width);
                let x_center = (content_width - self.frame().size.width) / 2.0;
                s.align_sent = (frame_origin.x - x_center).abs() <= 5.0;
                let _ = event;
            });
        }

        // IINA's `mouseDragged`: newOrigin = mouseLocation - mousePosRelatedToView,
        // stick to the horizontal centre within 5pt (one-shot alignment haptic),
        // bound to the window frame, then apply and report.
        #[unsafe(method(mouseDragged:))]
        fn mouse_dragged(&self, event: &NSEvent) {
            CONTROLS_STATE.with(|s| {
                let mut s = s.borrow_mut();
                let Some(mouse_pos) = s.mouse_pos else { return };
                let window = self.window();
                let Some(window) = window.as_ref() else { return };
                let _ = event;
                let screen_point = NSEvent::mouseLocation();
                let window_frame = window.frame();
                let p = NSPoint::new(
                    screen_point.x - window_frame.origin.x,
                    screen_point.y - window_frame.origin.y,
                );
                let content = window.contentView();
                let Some(content) = content.as_ref() else { return };
                let w = content.bounds().size.width;
                let h = content.bounds().size.height;
                let frame = self.frame();
                // IINA: newOrigin = currentLocation - mousePosRelatedToView
                let mut new_origin = NSPoint::new(p.x - mouse_pos.x, p.y - mouse_pos.y);
                // Stick to the horizontal centre (controlBarStickToCenter).
                let x_center = (w - frame.size.width) / 2.0;
                if (new_origin.x - x_center).abs() <= 5.0 {
                    new_origin.x = x_center;
                    if !s.align_sent {
                        snap_feedback();
                        s.align_sent = true;
                    }
                } else {
                    s.align_sent = false;
                }
                // IINA starts a drag at least 10pt from the leading edge. Its
                // trailing Auto Layout constraint is only a 1pt minimum, so
                // the effective trailing drag bound is 1pt rather than 10pt.
                let x_max = (w - frame.size.width - 1.0).max(0.0);
                let y_max = (h - frame.size.height - 25.0).max(0.0);
                new_origin.x = new_origin.x.clamp(10.0, x_max.max(10.0));
                new_origin.y = new_origin.y.clamp(0.0, y_max.max(0.0));
                self.setFrameOrigin(new_origin);
                // IINA persists the horizontal centre and bottom edge as
                // fractions of the video view, so resizing keeps the OSC in the
                // same relative position and updatePosition() can re-clamp it.
                if let Some(cb) = &s.callbacks {
                    let horizontal = if w > 0.0 {
                        ((new_origin.x + frame.size.width / 2.0) / w).clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    let vertical = if h > 0.0 {
                        (new_origin.y / h).clamp(0.0, 1.0)
                    } else {
                        0.1
                    };
                    (cb.on_move)(horizontal, vertical);
                }
            });
        }

        #[unsafe(method(mouseUp:))]
        fn mouse_up(&self, _event: &NSEvent) {
            CONTROLS_STATE.with(|s| s.borrow_mut().mouse_pos = None);
        }

        #[unsafe(method(kagamiSeek:))]
        fn kagami_seek(&self, slider: &NSSlider) {
            CONTROLS_STATE.with(|s| {
                if let Some(cb) = &s.borrow().callbacks {
                    (cb.on_seek)(slider.doubleValue());
                }
            });
        }

        #[unsafe(method(kagamiVolume:))]
        fn kagami_volume(&self, slider: &NSSlider) {
            CONTROLS_STATE.with(|s| {
                if let Some(cb) = &s.borrow().callbacks {
                    (cb.on_volume)(slider.doubleValue() as f32);
                }
            });
        }

        #[unsafe(method(kagamiPlay:))]
        fn kagami_play(&self, _sender: &AnyObject) {
            CONTROLS_STATE.with(|s| {
                if let Some(cb) = &s.borrow().callbacks {
                    (cb.on_play)();
                }
            });
        }

        #[unsafe(method(kagamiSeekBack:))]
        fn kagami_seek_back(&self, _sender: &AnyObject) {
            CONTROLS_STATE.with(|s| {
                if let Some(cb) = &s.borrow().callbacks {
                    (cb.on_seek_back)();
                }
            });
        }

        #[unsafe(method(kagamiSeekForward:))]
        fn kagami_seek_forward(&self, _sender: &AnyObject) {
            CONTROLS_STATE.with(|s| {
                if let Some(cb) = &s.borrow().callbacks {
                    (cb.on_seek_forward)();
                }
            });
        }

        #[unsafe(method(kagamiMute:))]
        fn kagami_mute(&self, _sender: &AnyObject) {
            CONTROLS_STATE.with(|s| {
                if let Some(cb) = &s.borrow().callbacks {
                    (cb.on_mute)();
                }
            });
        }

        #[unsafe(method(kagamiFullscreen:))]
        fn kagami_fullscreen(&self, _sender: &AnyObject) {
            CONTROLS_STATE.with(|s| {
                if let Some(cb) = &s.borrow().callbacks {
                    (cb.on_fullscreen)();
                }
            });
        }

        #[unsafe(method(kagamiSaveFrame:))]
        fn kagami_save_frame(&self, _sender: &AnyObject) {
            CONTROLS_STATE.with(|s| {
                if let Some(cb) = &s.borrow().callbacks {
                    (cb.on_save_frame)();
                }
            });
        }

        #[unsafe(method(kagamiAudioMenu:))]
        fn kagami_audio_menu(&self, sender: &NSButton) {
            // Clone before opening the menu: selection is synchronous and its
            // callback re-enters CONTROLS_STATE.
            let (tracks, current) = CONTROLS_STATE.with(|s| {
                let s = s.borrow();
                (s.audio_tracks.clone(), s.current_audio_track)
            });
            let mtm = self.mtm();
            let menu =
                NSMenu::initWithTitle(NSMenu::alloc(mtm), &NSString::from_str("Audio Track"));
            menu.setAutoenablesItems(false);
            let target = self as *const FloatingBar as *mut AnyObject;
            let mut bottom_item = None;

            for track in tracks {
                let item = unsafe {
                    NSMenuItem::initWithTitle_action_keyEquivalent(
                        NSMenuItem::alloc(mtm),
                        &NSString::from_str(&track.name),
                        Some(sel!(kagamiSelectAudioTrack:)),
                        &NSString::new(),
                    )
                };
                item.setTag(track.id as _);
                item.setState(if track.id == current {
                    NSControlStateValueOn
                } else {
                    NSControlStateValueOff
                });
                unsafe { item.setTarget(Some(&*target)) };
                menu.addItem(&item);
                bottom_item = Some(item);
            }

            // Anchor the menu's bottom row to the button instead of guessing
            // from `menu.size()`. AppKit lays every preceding row upward from
            // this item and handles label widths/screen clamping itself. The
            // one-row offset leaves the bottom row immediately above the icon.
            let Some(bottom_item) = bottom_item else { return };
            let menu_size = menu.size();
            let row_height = menu_size.height / menu.numberOfItems() as f64;
            menu.popUpMenuPositioningItem_atLocation_inView(
                Some(&bottom_item),
                NSPoint::new(
                    sender.bounds().size.width - menu_size.width,
                    sender.bounds().size.height + row_height,
                ),
                Some(sender),
            );
        }

        #[unsafe(method(kagamiSelectAudioTrack:))]
        fn kagami_select_audio_track(&self, item: &NSMenuItem) {
            CONTROLS_STATE.with(|s| {
                if let Some(cb) = &s.borrow().callbacks {
                    (cb.on_audio_track)(item.tag() as i32);
                }
            });
        }
    }
);

// ---------------------------------------------------------------------------
// IINA slider cells
// ---------------------------------------------------------------------------

fn is_dark_appearance() -> bool {
    NSAppearance::currentDrawingAppearance()
        .name()
        .to_string()
        .contains("Dark")
}

fn slider_bar_color(dark: bool, dark_alpha: f64, light_alpha: f64) -> Retained<NSColor> {
    if dark {
        NSColor::colorWithSRGBRed_green_blue_alpha(1.0, 1.0, 1.0, dark_alpha)
    } else {
        NSColor::colorWithSRGBRed_green_blue_alpha(0.0, 0.0, 0.0, light_alpha)
    }
}

define_class!(
    #[unsafe(super(NSSliderCell))]
    #[thread_kind = MainThreadOnly]
    #[name = "KagamiPlaySliderCell"]
    struct PlaySliderCell;

    impl PlaySliderCell {
        // IINA's PlaySliderCell uses a 3pt wide, 15pt tall rounded knob. The
        // wider track keeps the playhead visible over video without being as
        // loud as AppKit's default slider.
        #[unsafe(method(knobThickness))]
        fn knob_thickness(&self) -> f64 {
            3.0
        }

        // Match IINA's custom knob placement as well as its drawing. AppKit's
        // stock cell leaves a different usable track width; IINA positions the
        // 3pt playhead over `barWidth - knobWidth` so 0% and 100% land exactly
        // at the two ends of the rounded bar.
        #[unsafe(method(knobRectFlipped:))]
        fn knob_rect(&self, flipped: bool) -> NSRect {
            let bar_rect = self.barRectFlipped(flipped);
            let range = self.maxValue() - self.minValue();
            let percentage = if range == 0.0 {
                0.0
            } else {
                self.doubleValue() / range
            };
            let effective_bar_width = bar_rect.size.width - 3.0;
            let pos = bar_rect.origin.x + percentage * effective_bar_width;
            let native_rect: NSRect = unsafe { msg_send![super(self), knobRectFlipped: flipped] };
            let height = (bar_rect.origin.y - native_rect.origin.y) * 2.0 + bar_rect.size.height;
            rect(pos, native_rect.origin.y, 3.0, height)
        }

        #[unsafe(method(drawBarInside:flipped:))]
        fn draw_bar(&self, rect: NSRect, flipped: bool) {
            let bar_rect = rect;
            let path = NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(bar_rect, 1.5, 1.5);
            let knob = self.knobRectFlipped(flipped);
            let knob_x = knob.origin.x.round();
            let dark = is_dark_appearance();
            let gap_left = if dark { knob_x - 1.0 } else { knob_x };
            let gap_right = if dark {
                knob_x + knob.size.width + 1.0
            } else {
                knob_x
            };

            NSGraphicsContext::saveGraphicsState_class();
            let left = NSBezierPath::bezierPathWithRect(NSRect::new(
                NSPoint::new(bar_rect.origin.x, bar_rect.origin.y),
                NSSize::new(
                    (gap_left - bar_rect.origin.x).max(0.0),
                    bar_rect.size.height,
                ),
            ));
            left.addClip();
            // IINA MainSliderBarLeft: black 40% in light, white 30% in dark.
            slider_bar_color(dark, 0.30, 0.40).setFill();
            path.fill();
            NSGraphicsContext::restoreGraphicsState_class();

            NSGraphicsContext::saveGraphicsState_class();
            let right = NSBezierPath::bezierPathWithRect(NSRect::new(
                NSPoint::new(gap_right, bar_rect.origin.y),
                NSSize::new(
                    (bar_rect.origin.x + bar_rect.size.width - gap_right).max(0.0),
                    bar_rect.size.height,
                ),
            ));
            right.addClip();
            // IINA MainSliderBarRight: black 20% in light, white 10% in dark.
            slider_bar_color(dark, 0.10, 0.20).setFill();
            path.fill();
            NSGraphicsContext::restoreGraphicsState_class();
        }

        #[unsafe(method(drawKnob:))]
        fn draw_knob(&self, knob_rect: NSRect) {
            let rect = NSRect::new(
                NSPoint::new(
                    knob_rect.origin.x.round(),
                    knob_rect.origin.y + 0.5 * (knob_rect.size.height - 15.0),
                ),
                NSSize::new(knob_rect.size.width, 15.0),
            );
            let path = NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(rect, 1.0, 1.0);
            let dark = is_dark_appearance();
            // IINA MainSliderKnob / MainSliderKnobActive asset alphas.
            let alpha = match (dark, self.isHighlighted()) {
                (false, false) => 0.96,
                (false, true) => 0.75,
                (true, false) => 0.80,
                (true, true) => 0.98,
            };
            if !dark {
                NSGraphicsContext::saveGraphicsState_class();
                let shadow = NSShadow::new();
                shadow.setShadowBlurRadius(1.0);
                shadow.setShadowOffset(NSSize::new(0.0, -0.5));
                shadow.set();
                NSColor::colorWithSRGBRed_green_blue_alpha(1.0, 1.0, 1.0, alpha).setFill();
                path.fill();
                path.setLineWidth(0.4);
                NSColor::colorWithSRGBRed_green_blue_alpha(0.0, 0.0, 0.0, 1.0 / 3.0)
                    .setStroke();
                path.stroke();
                NSGraphicsContext::restoreGraphicsState_class();
                return;
            }
            NSColor::colorWithSRGBRed_green_blue_alpha(1.0, 1.0, 1.0, alpha).setFill();
            path.fill();
        }
    }
);

define_class!(
    #[unsafe(super(NSSliderCell))]
    #[thread_kind = MainThreadOnly]
    #[name = "KagamiVolumeSliderCell"]
    struct VolumeSliderCell;

    impl VolumeSliderCell {
        #[unsafe(method(drawBarInside:flipped:))]
        fn draw_bar(&self, rect: NSRect, flipped: bool) {
            let knob = self.knobRectFlipped(flipped).origin.x.round();
            let radius = 1.5;
            let path = NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(rect, radius, radius);
            let dark = is_dark_appearance();

            NSGraphicsContext::saveGraphicsState_class();
            let left = NSBezierPath::bezierPathWithRect(NSRect::new(
                NSPoint::new(rect.origin.x, rect.origin.y),
                NSSize::new((knob - rect.origin.x).max(0.0), rect.size.height),
            ));
            left.addClip();
            // IINA VolumeSliderBarLeft: black 40% in light, white 30% in dark.
            slider_bar_color(dark, 0.30, 0.40).setFill();
            path.fill();
            NSGraphicsContext::restoreGraphicsState_class();

            NSGraphicsContext::saveGraphicsState_class();
            let right = NSBezierPath::bezierPathWithRect(NSRect::new(
                NSPoint::new(knob, rect.origin.y),
                NSSize::new(
                    (rect.origin.x + rect.size.width - knob).max(0.0),
                    rect.size.height,
                ),
            ));
            right.addClip();
            // IINA VolumeSliderBarRight: 10% in both appearances.
            slider_bar_color(dark, 0.10, 0.10).setFill();
            path.fill();
            NSGraphicsContext::restoreGraphicsState_class();
        }
    }
);

// ---------------------------------------------------------------------------
// VideoState
// ---------------------------------------------------------------------------

pub struct VideoState<'a> {
    pub position: f64,
    pub duration: f64,
    pub paused: bool,
    pub muted: bool,
    pub volume: f32,
    pub audio_tracks: &'a [crate::video::AudioTrack],
    pub current_audio_track: i32,
    pub visible: bool,
}

// ---------------------------------------------------------------------------
// The native panel
// ---------------------------------------------------------------------------

pub struct NativeControls {
    mtm: MainThreadMarker,
    container: Retained<ControlsOverlay>,
    bar: Retained<FloatingBar>,
    backdrop: Backdrop,
    content: Retained<ControlsOverlay>,
    speaker: Retained<VolumeButton>,
    speaker_image: Retained<NSImageView>,
    volume: Retained<NSSlider>,
    left_arrow: Retained<NSButton>,
    play: Retained<NSButton>,
    right_arrow: Retained<NSButton>,
    toolbar_plugins: Retained<NSButton>,
    toolbar_pip: Retained<NSButton>,
    toolbar_playlist: Retained<NSButton>,
    toolbar_settings: Retained<NSButton>,
    toolbar_audio: Retained<NSButton>,
    elapsed: Retained<NSTextField>,
    seek: Retained<NSSlider>,
    duration: Retained<NSTextField>,
    last_paused: Cell<Option<bool>>,
    last_muted: Cell<Option<bool>>,
    last_volume_level: Cell<Option<f64>>,
}

// IINA still uses a historical 67pt value in updatePosition(), but the current
// AppKit constraints render the default (non-compact) floating OSC at 82pt:
// 14 top + 24 controls + 8 row spacing + 28 timeline alignment height + 8
// bottom. The timeline's actual view frame is 34pt because its custom class has
// a 6pt top alignment inset. Keep the two values separate so both appearance
// and IINA's positioning behavior match the source.
const BAR_H: f64 = 82.0;
const IINA_POSITION_H: f64 = 67.0;
const IINA_BAR_W: f64 = 460.0;
const BAR_MAX_W: f64 = IINA_BAR_W + ICON_H;
const BAR_SIDE_PADDING: f64 = 1.0;
const BAR_MIN_W: f64 = 200.0;
const TOP_PADDING: f64 = 14.0;
const TOP_HORIZONTAL_PADDING: f64 = 12.0;
const BOTTOM_PADDING: f64 = 8.0;
const BOTTOM_HORIZONTAL_PADDING: f64 = 8.0;
const ICON_H: f64 = 24.0;
const VOL_W: f64 = 70.0;
const VOLUME_FRAME_W: f64 = 74.0;
const VOLUME_FRAME_H: f64 = 17.0;
const PLAY_SLIDER_FRAME_H: f64 = 20.0;
const LABEL_FRAME_MIN_W: f64 = 50.0;
const LABEL_H: f64 = 14.0;
const GROUP_SPACING: f64 = 24.0;
// IINA keeps a 30pt speed-label container on each side of the three 24pt
// transport buttons even when the labels themselves are hidden. Those blank
// containers are part of the centre group's layout footprint and therefore of
// the width at which AppKit detaches the volume and toolbar groups.
const PLAY_CONTROL_SIDE_RESERVE: f64 = 30.0;

fn rect(x: f64, y: f64, w: f64, h: f64) -> NSRect {
    NSRect::new(NSPoint::new(x, y), NSSize::new(w, h))
}

fn symbol(name: &str) -> Option<Retained<NSImage>> {
    let s = NSString::from_str(name);
    NSImage::imageWithSystemSymbolName_accessibilityDescription(&s, None)
}

fn symbol_configuration(
    point_size: f64,
    weight: objc2_app_kit::NSFontWeight,
) -> Retained<NSImageSymbolConfiguration> {
    NSImageSymbolConfiguration::configurationWithPointSize_weight(point_size, weight)
}

fn configured_symbol(
    name: &str,
    point_size: f64,
    weight: objc2_app_kit::NSFontWeight,
) -> Option<Retained<NSImage>> {
    let image = symbol(name)?;
    let configuration = symbol_configuration(point_size, weight);
    image.imageWithSymbolConfiguration(&configuration)
}

#[derive(Clone, Copy)]
enum TransportIcon {
    Play,
    Pause,
    SpeedLeft,
    SpeedRight,
}

fn cubic(path: &NSBezierPath, end: (f64, f64), control1: (f64, f64), control2: (f64, f64)) {
    path.curveToPoint_controlPoint1_controlPoint2(
        NSPoint::new(end.0, end.1),
        NSPoint::new(control1.0, control1.1),
        NSPoint::new(control2.0, control2.1),
    );
}

fn draw_play_path(path: &NSBezierPath) {
    path.moveToPoint(NSPoint::new(22.3007, 12.8866));
    cubic(path, (22.8381, 12.0), (22.631, 12.7142), (22.8381, 12.3726));
    cubic(
        path,
        (22.3007, 11.1134),
        (22.8381, 11.6274),
        (22.631, 11.2858),
    );
    cubic(path, (2.4626, 0.7631), (17.9618, 8.8496), (6.5301, 2.8853));
    cubic(path, (1.4816, 0.7945), (2.1526, 0.6013), (1.7806, 0.6133));
    cubic(path, (1.0, 1.6497), (1.1826, 0.9758), (1.0, 1.3));
    cubic(path, (1.0, 22.3503), (1.0, 6.0379), (1.0, 17.9621));
    cubic(path, (1.4816, 23.2055), (1.0, 22.7), (1.1826, 23.0242));
    cubic(
        path,
        (2.4626, 23.2369),
        (1.7806, 23.3867),
        (2.1526, 23.3987),
    );
    cubic(
        path,
        (22.3007, 12.8866),
        (6.5301, 21.1147),
        (17.9618, 15.1504),
    );
    path.closePath();
}

fn draw_pause_bar(path: &NSBezierPath, x0: f64, x1: f64) {
    path.moveToPoint(NSPoint::new(x1, 22.0));
    cubic(path, (x1 - 1.0, 23.0), (x1, 22.5523), (x1 - 0.4477, 23.0));
    cubic(
        path,
        (x0 + 1.0, 23.0),
        (x1 - 2.537, 23.0),
        (x0 + 2.537, 23.0),
    );
    cubic(path, (x0, 22.0), (x0 + 0.4477, 23.0), (x0, 22.5523));
    cubic(path, (x0, 2.0), (x0, 18.3354), (x0, 5.6646));
    cubic(path, (x0 + 1.0, 1.0), (x0, 1.4477), (x0 + 0.4477, 1.0));
    cubic(path, (x1 - 1.0, 1.0), (x0 + 2.537, 1.0), (x1 - 2.537, 1.0));
    cubic(path, (x1, 2.0), (x1 - 0.4477, 1.0), (x1, 1.4477));
    cubic(path, (x1, 22.0), (x1, 5.6646), (x1, 18.3354));
    path.closePath();
}

fn draw_speed_left_triangle(path: &NSBezierPath, offset_x: f64) {
    // Exact path from IINA's 24x24 `speedl.pdf` asset. The previous
    // transcription started at an unrelated point near the upper-left corner,
    // which folded the closing segment across the triangle and made the back
    // glyph look visibly malformed.
    path.moveToPoint(NSPoint::new(offset_x + 1.2481, 11.1679));
    cubic(
        path,
        (offset_x + 0.8028, 12.0),
        (offset_x + 0.9699, 11.3534),
        (offset_x + 0.8028, 11.6656),
    );
    cubic(
        path,
        (offset_x + 1.2481, 12.8321),
        (offset_x + 0.8028, 12.3344),
        (offset_x + 0.9699, 12.6466),
    );
    cubic(
        path,
        (offset_x + 10.4453, 18.9635),
        (offset_x + 3.4544, 14.3029),
        (offset_x + 8.0374, 17.3583),
    );
    cubic(
        path,
        (offset_x + 11.4719, 19.0132),
        (offset_x + 10.7522, 19.1681),
        (offset_x + 11.1467, 19.1872),
    );
    cubic(
        path,
        (offset_x + 12.0, 18.1315),
        (offset_x + 11.797, 18.8391),
        (offset_x + 12.0, 18.5003),
    );
    cubic(
        path,
        (offset_x + 12.0, 5.8685),
        (offset_x + 12.0, 15.0524),
        (offset_x + 12.0, 8.9476),
    );
    cubic(
        path,
        (offset_x + 11.4719, 4.9868),
        (offset_x + 12.0, 5.4997),
        (offset_x + 11.797, 5.1609),
    );
    cubic(
        path,
        (offset_x + 10.4453, 5.0365),
        (offset_x + 11.1467, 4.8128),
        (offset_x + 10.7522, 4.8319),
    );
    cubic(
        path,
        (offset_x + 1.2481, 11.1679),
        (offset_x + 8.0374, 6.6417),
        (offset_x + 3.4544, 9.6971),
    );
    path.closePath();
}

fn draw_speed_right_triangle(path: &NSBezierPath, offset_x: f64) {
    path.moveToPoint(NSPoint::new(offset_x + 10.7519, 12.8321));
    cubic(
        path,
        (offset_x + 11.1972, 12.0),
        (offset_x + 11.0301, 12.6466),
        (offset_x + 11.1972, 12.3344),
    );
    cubic(
        path,
        (offset_x + 10.7519, 11.1679),
        (offset_x + 11.1972, 11.6656),
        (offset_x + 11.0301, 11.3534),
    );
    cubic(
        path,
        (offset_x + 1.5547, 5.0365),
        (offset_x + 8.5456, 9.6971),
        (offset_x + 3.9626, 6.6417),
    );
    cubic(
        path,
        (offset_x + 0.5281, 4.9868),
        (offset_x + 1.2478, 4.8319),
        (offset_x + 0.8533, 4.8128),
    );
    cubic(
        path,
        (offset_x, 5.8685),
        (offset_x + 0.203, 5.1609),
        (offset_x - 0.0001, 5.4997),
    );
    cubic(
        path,
        (offset_x, 18.1315),
        (offset_x, 8.9476),
        (offset_x, 15.0524),
    );
    cubic(
        path,
        (offset_x + 0.5281, 19.0132),
        (offset_x - 0.0001, 18.5003),
        (offset_x + 0.203, 18.8391),
    );
    cubic(
        path,
        (offset_x + 1.5547, 18.9635),
        (offset_x + 0.8533, 19.1872),
        (offset_x + 1.2478, 19.1681),
    );
    cubic(
        path,
        (offset_x + 10.7519, 12.8321),
        (offset_x + 3.9626, 17.3583),
        (offset_x + 8.5456, 14.3029),
    );
    path.closePath();
}

fn transport_image(_mtm: MainThreadMarker, icon: TransportIcon) -> Retained<NSImage> {
    // IINA ships these as 24x24 template PDF assets. Use AppKit's
    // resolution-independent drawing handler so our source-matched Bezier paths
    // retain the same vector behavior instead of baking a 1x bitmap with
    // `lockFocus`.
    let drawing = RcBlock::new(move |_dst_rect: NSRect| -> Bool {
        NSColor::blackColor().setFill();
        match icon {
            TransportIcon::Play => {
                let path = NSBezierPath::bezierPath();
                draw_play_path(&path);
                path.fill();
            }
            TransportIcon::Pause => {
                let left = NSBezierPath::bezierPath();
                draw_pause_bar(&left, 2.0, 10.0);
                left.fill();
                let right = NSBezierPath::bezierPath();
                draw_pause_bar(&right, 14.0, 22.0);
                right.fill();
            }
            TransportIcon::SpeedLeft => {
                let left = NSBezierPath::bezierPath();
                draw_speed_left_triangle(&left, 0.0);
                left.fill();
                let right = NSBezierPath::bezierPath();
                draw_speed_left_triangle(&right, 12.0);
                right.fill();
            }
            TransportIcon::SpeedRight => {
                let left = NSBezierPath::bezierPath();
                draw_speed_right_triangle(&left, 0.0);
                left.fill();
                let right = NSBezierPath::bezierPath();
                draw_speed_right_triangle(&right, 12.0);
                right.fill();
            }
        }
        Bool::YES
    });
    let image =
        NSImage::imageWithSize_flipped_drawingHandler(NSSize::new(24.0, 24.0), false, &drawing);
    image.setTemplate(true);
    image
}

fn make_button(
    mtm: MainThreadMarker,
    name: &str,
    fallback: Option<&str>,
    point_size: f64,
) -> Retained<NSButton> {
    let b = NSButton::new(mtm);
    if let Some(img) = symbol(name).or_else(|| fallback.and_then(symbol)) {
        b.setImage(Some(&img));
    }
    let configuration = symbol_configuration(point_size, unsafe { NSFontWeightMedium });
    b.setSymbolConfiguration(Some(&configuration));
    b.setImagePosition(objc2_app_kit::NSCellImagePosition::ImageOnly);
    b.setImageScaling(NSImageScaling::ScaleProportionallyDown);
    b.setContentTintColor(Some(&NSColor::labelColor()));
    b.setButtonType(NSButtonType::MomentaryLight);
    b.setBezelStyle(NSBezelStyle::SmallSquare);
    b.setBordered(false);
    b.setRefusesFirstResponder(true);
    b
}

// IINA's current OSCToolbarButton deliberately uses `.regularSquare` for these
// borderless 24pt buttons. AppKit marks that bezel constant deprecated on newer
// SDKs, but replacing it would change the cell metrics we are matching.
#[allow(deprecated)]
fn set_iina_toolbar_bezel(button: &NSButton) {
    button.setBezelStyle(NSBezelStyle::RegularSquare);
}

enum Backdrop {
    Glass(Retained<GlassBackdrop>),
    Legacy(Retained<VisualEffectBackdrop>),
}

impl Backdrop {
    fn set_frame(&self, frame: NSRect) {
        match self {
            Self::Glass(view) => view.setFrame(frame),
            Self::Legacy(view) => view.setFrame(frame),
        }
    }

    fn add_to_bar(&self, bar: &FloatingBar) {
        match self {
            Self::Glass(view) => bar.addSubview(view),
            Self::Legacy(view) => bar.addSubview(view),
        }
    }

    fn set_content(&self, content: &ControlsOverlay) {
        match self {
            // This is the same containment used by IINA's TranslucentView.
            Self::Glass(view) => view.setContentView(Some(content)),
            Self::Legacy(view) => view.addSubview(content),
        }
    }
}

fn supports_liquid_glass() -> bool {
    AnyClass::get(c"NSGlassEffectView").is_some()
}

fn install_cell(slider: &NSSlider, cell: &NSSliderCell) {
    slider.setCell(Some(cell));
}

/// A short haptic "tick" (the alignment pattern), used to confirm the panel
/// stuck to the centre. IINA uses `.alignment` here.
pub fn snap_feedback() {
    let performer = NSHapticFeedbackManager::defaultPerformer();
    performer.performFeedbackPattern_performanceTime(
        NSHapticFeedbackPattern::Alignment,
        NSHapticFeedbackPerformanceTime::Default,
    );
}

impl NativeControls {
    pub fn new(mtm: MainThreadMarker) -> Option<Self> {
        let app = NSApplication::sharedApplication(mtm);
        let window = app.keyWindow().or_else(|| app.mainWindow())?;
        let window_content = window.contentView()?;
        let bounds = window_content.bounds();

        // A full-window transparent container. Only its bar subview receives
        // AppKit mouse events; everything outside the bar still falls through to
        // egui/wgpu below.
        let container: Retained<ControlsOverlay> = {
            let this = ControlsOverlay::alloc(mtm);
            unsafe { msg_send![this, initWithFrame: bounds] }
        };

        let bar: Retained<FloatingBar> = {
            let this = FloatingBar::alloc(mtm);
            unsafe { msg_send![this, initWithFrame: bounds] }
        };

        // IINA's `TranslucentView`: Liquid Glass with a 12pt radius on macOS 26,
        // otherwise the legacy popover visual effect clipped to a 6pt radius.
        let liquid_glass = supports_liquid_glass();
        let backdrop = if liquid_glass {
            let v: Retained<GlassBackdrop> = {
                let this = GlassBackdrop::alloc(mtm);
                unsafe { msg_send![this, initWithFrame: bounds] }
            };
            v.setCornerRadius(12.0);
            Backdrop::Glass(v)
        } else {
            let v: Retained<VisualEffectBackdrop> = {
                let this = VisualEffectBackdrop::alloc(mtm);
                unsafe { msg_send![this, initWithFrame: bounds] }
            };
            v.setMaterial(NSVisualEffectMaterial::Popover);
            v.setBlendingMode(NSVisualEffectBlendingMode::WithinWindow);
            v.setState(NSVisualEffectState::Active);
            v.setClipsToBounds(true);
            v.setWantsLayer(true);
            unsafe {
                let layer: *mut AnyObject = msg_send![&*v, layer];
                if !layer.is_null() {
                    let _: () = msg_send![layer, setCornerRadius: 6.0_f64];
                    let _: () = msg_send![layer, setMasksToBounds: true];
                }
            }
            Backdrop::Legacy(v)
        };
        backdrop.add_to_bar(&bar);

        // Keep the controls inside the glass/frosted container, matching
        // TranslucentView's native hierarchy rather than drawing them as sibling
        // views above the material.
        let osc_content: Retained<ControlsOverlay> = {
            let this = ControlsOverlay::alloc(mtm);
            unsafe { msg_send![this, initWithFrame: bounds] }
        };
        backdrop.set_content(&osc_content);

        let speaker: Retained<VolumeButton> = {
            let this = VolumeButton::alloc(mtm);
            unsafe { msg_send![this, initWithFrame: rect(0.0, 0.0, ICON_H, ICON_H)] }
        };
        let speaker_image = NSImageView::new(mtm);
        speaker_image.setImageScaling(NSImageScaling::ScaleProportionallyDown);
        if let Some(img) =
            configured_symbol("speaker.wave.2.fill", 13.0, unsafe { NSFontWeightRegular })
        {
            speaker_image.setImage(Some(&img));
        }
        speaker.addSubview(&speaker_image);
        let left_arrow = make_button(mtm, "backward.fill", None, 14.0);
        left_arrow.setImage(Some(&transport_image(mtm, TransportIcon::SpeedLeft)));
        let play = make_button(mtm, "play.fill", None, 14.0);
        play.setImage(Some(&transport_image(mtm, TransportIcon::Play)));
        let right_arrow = make_button(mtm, "forward.fill", None, 14.0);
        right_arrow.setImage(Some(&transport_image(mtm, TransportIcon::SpeedRight)));

        // Keep IINA's toolbar controls and add audio-track selection at the end.
        // Every button occupies 24x24 points with zero spacing.
        let toolbar_plugins = make_button(mtm, "puzzlepiece.extension", Some("puzzlepiece"), 13.5);
        let toolbar_pip = make_button(mtm, "pip.enter", None, 13.5);
        let toolbar_playlist = make_button(mtm, "list.bullet.rectangle", Some("list.bullet"), 14.0);
        let toolbar_settings = make_button(mtm, "gearshape", None, 14.0);
        let toolbar_audio = make_button(mtm, "waveform", Some("speaker.wave.2.fill"), 14.0);
        set_iina_toolbar_bezel(&toolbar_plugins);
        set_iina_toolbar_bezel(&toolbar_pip);
        set_iina_toolbar_bezel(&toolbar_playlist);
        set_iina_toolbar_bezel(&toolbar_settings);
        set_iina_toolbar_bezel(&toolbar_audio);
        toolbar_audio.setToolTip(Some(&NSString::from_str("Audio Track")));

        let time_font = NSFont::messageFontOfSize(11.0);
        let make_label = || {
            let l = NSTextField::labelWithString(&NSString::from_str("0:00"), mtm);
            l.setControlSize(NSControlSize::Mini);
            l.setTextColor(Some(&NSColor::secondaryLabelColor()));
            l.setAlignment(NSTextAlignment::Center);
            l.setFont(Some(&time_font));
            l
        };
        let elapsed = make_label();
        let duration = make_label();

        let make_slider = || {
            let s = NSSlider::new(mtm);
            s.setMinValue(0.0);
            s.setMaxValue(1.0);
            s.setRefusesFirstResponder(true);
            s
        };
        let seek = make_slider();
        let volume = make_slider();
        // A seek restarts VLC's decoder. AppKit's continuous slider action can
        // therefore turn one physical click/drag into several real seeks at
        // slightly different timestamps, making the first decoded chunk after
        // the target appear to play twice. Let NSSlider track its knob locally
        // during the gesture and commit exactly once on mouse-up instead. A
        // click still lands immediately when released, while avoiding redundant
        // decoder restarts entirely. Volume has no such cost and stays live.
        seek.setContinuous(false);
        volume.setContinuous(true);
        let seek_cell: Retained<PlaySliderCell> = {
            let this = PlaySliderCell::alloc(mtm);
            unsafe { msg_send![this, init] }
        };
        let volume_cell: Retained<VolumeSliderCell> = {
            let this = VolumeSliderCell::alloc(mtm);
            unsafe { msg_send![this, init] }
        };
        install_cell(&seek, &seek_cell);
        install_cell(&volume, &volume_cell);
        // Apply control sizes after replacing the cells. NSSlider forwards
        // `setControlSize:` to its current cell; doing this before `setCell:`
        // leaves the replacement VolumeSliderCell at AppKit's regular size,
        // producing the oversized knob. This matches IINA's construction order:
        // VolumeSlider installs its cell, then PlayerWindowController makes the
        // slider `.mini`, retaining AppKit's native mini knob appearance.
        seek.setControlSize(NSControlSize::Small);
        volume.setControlSize(NSControlSize::Mini);

        osc_content.addSubview(&speaker);
        osc_content.addSubview(&volume);
        osc_content.addSubview(&left_arrow);
        osc_content.addSubview(&play);
        osc_content.addSubview(&right_arrow);
        osc_content.addSubview(&toolbar_plugins);
        osc_content.addSubview(&toolbar_pip);
        osc_content.addSubview(&toolbar_playlist);
        osc_content.addSubview(&toolbar_settings);
        osc_content.addSubview(&toolbar_audio);
        osc_content.addSubview(&elapsed);
        osc_content.addSubview(&seek);
        osc_content.addSubview(&duration);

        container.addSubview(&bar);
        window_content.addSubview(&container);

        Some(Self {
            mtm,
            container,
            bar,
            backdrop,
            content: osc_content,
            speaker,
            speaker_image,
            volume,
            left_arrow,
            play,
            right_arrow,
            toolbar_plugins,
            toolbar_pip,
            toolbar_playlist,
            toolbar_settings,
            toolbar_audio,
            elapsed,
            seek,
            duration,
            last_paused: Cell::new(None),
            last_muted: Cell::new(None),
            last_volume_level: Cell::new(None),
        })
    }

    /// Install or replace the callbacks that fire when the native controls are
    /// used. Must be called on the main thread.
    pub fn set_callbacks(&self, cb: ControlsCallbacks) {
        CONTROLS_STATE.with(|s| s.borrow_mut().callbacks = Some(cb));
        let target = self.bar.as_ref() as *const FloatingBar as *mut AnyObject;
        unsafe {
            let target = &*target;
            self.seek.setTarget(Some(target));
            self.seek.setAction(Some(sel!(kagamiSeek:)));
            self.volume.setTarget(Some(target));
            self.volume.setAction(Some(sel!(kagamiVolume:)));
            self.left_arrow.setTarget(Some(target));
            self.left_arrow.setAction(Some(sel!(kagamiSeekBack:)));
            self.play.setTarget(Some(target));
            self.play.setAction(Some(sel!(kagamiPlay:)));
            self.right_arrow.setTarget(Some(target));
            self.right_arrow.setAction(Some(sel!(kagamiSeekForward:)));
            self.toolbar_audio.setTarget(Some(target));
            self.toolbar_audio.setAction(Some(sel!(kagamiAudioMenu:)));
        }
    }

    pub fn hide(&self) {
        self.container.setHidden(true);
    }

    /// Lay the bar out for a `w`x`h` (points) content view and push the current
    /// playback state into the widgets. `pos` uses IINA's persisted normalized
    /// horizontal/vertical OSC coordinates; `None` restores IINA's 0.5 / 0.1
    /// defaults.
    pub fn update(
        &self,
        w: f64,
        h: f64,
        st: &VideoState<'_>,
        elapsed: &str,
        duration: &str,
        pos: Option<(f32, f32)>,
    ) {
        self.container.setFrame(rect(0.0, 0.0, w, h));
        CONTROLS_STATE.with(|s| {
            let mut s = s.borrow_mut();
            if s.audio_tracks != st.audio_tracks {
                s.audio_tracks.clear();
                s.audio_tracks.extend_from_slice(st.audio_tracks);
            }
            s.current_audio_track = st.current_audio_track;
        });
        self.toolbar_audio.setEnabled(!st.audio_tracks.is_empty());
        // Native controls can consume the mouse events that egui normally uses
        // to refresh `controls_until`. If that timer expires during a slider or
        // panel drag, hiding this container cancels the gesture and makes the
        // OSC appear to vanish under the pointer. Once the panel is visible,
        // keep it visible for the duration of the primary-button gesture. A
        // hidden panel stays hidden when the user clicks elsewhere in the app.
        let primary_mouse_down = NSEvent::pressedMouseButtons() & 1 != 0;
        let keep_visible_for_gesture = !self.container.isHidden() && primary_mouse_down;
        let visible = st.visible || keep_visible_for_gesture;
        self.container.setHidden(!visible);
        if !visible {
            return;
        }

        // The source OSC is 460pt wide; the added audio button extends its
        // trailing toolbar by one 24pt slot. Keep the same 1pt side padding and
        // narrow-window behavior.
        let bw = (w - 2.0 * BAR_SIDE_PADDING)
            .clamp(BAR_MIN_W, BAR_MAX_W)
            .min(w.max(0.0));
        let (horizontal, vertical) = pos.map(|(x, y)| (x as f64, y as f64)).unwrap_or((0.5, 0.1));
        // Once the full OSC no longer fits, force its X position to the video
        // centre. At full width, preserve the normalized position while
        // clamping the complete OSC into the view.
        let centre_x = if w < BAR_MAX_W {
            w / 2.0
        } else {
            let min_center = bw / 2.0 + BAR_SIDE_PADDING;
            let max_center = (w - bw / 2.0 - BAR_SIDE_PADDING).max(min_center);
            (w * horizontal).clamp(min_center, max_center)
        };
        let bx = (centre_x - bw / 2.0).max(0.0);
        let by = (h * vertical).clamp(0.0, (h - IINA_POSITION_H - 25.0).max(0.0));
        self.bar.setFrame(rect(bx, by, bw, BAR_H));
        // Subviews of the bar use the bar's local coordinate system. Keeping
        // window-space bx/by here used to apply the bar's offset twice.
        self.backdrop.set_frame(rect(0.0, 0.0, bw, BAR_H));
        self.content.setFrame(rect(0.0, 0.0, bw, BAR_H));

        // Top transport row. IINA uses a horizontal stack, so the volume group
        // and toolbar group stay pinned to their edges while the play/seek
        // group remains exactly centred.
        // IINA's LayoutValue(14, 10) / LayoutValue(8, 5) are normal-vs-compact
        // UI values, not Liquid Glass-vs-visual-effect values. Kagami has no
        // compact-UI preference, so match IINA's default (normal) geometry.
        let top_y = BAR_H - TOP_PADDING - ICON_H;
        self.speaker
            .setFrame(rect(TOP_HORIZONTAL_PADDING + 4.0, top_y, ICON_H, ICON_H));
        self.speaker_image.setFrame(rect(0.0, 0.0, ICON_H, ICON_H));
        // The volume slider's 70x13 alignment rect has NSSlider's native 2pt
        // alignment insets on all sides, so its actual frame is 74x17 and starts
        // 2pt earlier than the alignment-anchor math suggests.
        self.volume.setFrame(rect(
            TOP_HORIZONTAL_PADDING + 30.0,
            top_y + 4.0,
            VOLUME_FRAME_W,
            VOLUME_FRAME_H,
        ));

        let group_w = ICON_H * 3.0 + GROUP_SPACING * 2.0;
        let group_x = (bw - group_w) / 2.0;
        self.left_arrow
            .setFrame(rect(group_x, top_y, ICON_H, ICON_H));
        self.play.setFrame(rect(
            group_x + ICON_H + GROUP_SPACING,
            top_y,
            ICON_H,
            ICON_H,
        ));
        self.right_arrow.setFrame(rect(
            group_x + (ICON_H + GROUP_SPACING) * 2.0,
            top_y,
            ICON_H,
            ICON_H,
        ));

        // IINA lets the leading volume group and trailing toolbar detach when
        // the floating OSC gets too narrow, while the transport controls stay
        // centered. Its macOS 11 fallback spells out the threshold exactly:
        // play controls + 2 * max(volume, toolbar) + 2 * (10 + 12).
        let volume_group_w = 4.0 + ICON_H + 4.0 + VOL_W + 6.0;
        let toolbar_group_w = ICON_H * 5.0;
        let play_control_layout_w = group_w + PLAY_CONTROL_SIDE_RESERVE * 2.0;
        let side_group_threshold =
            play_control_layout_w + 2.0 * volume_group_w.max(toolbar_group_w) + 2.0 * (10.0 + 12.0);
        let show_side_groups = bw >= side_group_threshold;
        self.speaker.setHidden(!show_side_groups);
        self.volume.setHidden(!show_side_groups);
        self.toolbar_plugins.setHidden(!show_side_groups);
        self.toolbar_pip.setHidden(!show_side_groups);
        self.toolbar_playlist.setHidden(!show_side_groups);
        self.toolbar_settings.setHidden(!show_side_groups);
        self.toolbar_audio.setHidden(!show_side_groups);

        let toolbar_x = bw - TOP_HORIZONTAL_PADDING - ICON_H * 5.0;
        self.toolbar_plugins
            .setFrame(rect(toolbar_x, top_y, ICON_H, ICON_H));
        self.toolbar_pip
            .setFrame(rect(toolbar_x + ICON_H, top_y, ICON_H, ICON_H));
        self.toolbar_playlist
            .setFrame(rect(toolbar_x + ICON_H * 2.0, top_y, ICON_H, ICON_H));
        self.toolbar_settings
            .setFrame(rect(toolbar_x + ICON_H * 3.0, top_y, ICON_H, ICON_H));
        self.toolbar_audio
            .setFrame(rect(toolbar_x + ICON_H * 4.0, top_y, ICON_H, ICON_H));

        // Bottom slider row. The source constrains each NSTextField's alignment
        // rectangle to >=46pt. NSTextField adds 2pt alignment insets left/right,
        // so the actual minimum frame is 50pt. Likewise the small NSSlider has a
        // 16pt alignment height but a 20pt frame. These frame values were
        // verified against AppKit's own constraint solver for the IINA layout.
        self.elapsed.setStringValue(&NSString::from_str(elapsed));
        self.duration.setStringValue(&NSString::from_str(duration));
        let elapsed_w = self.elapsed.fittingSize().width.max(LABEL_FRAME_MIN_W);
        let duration_w = self.duration.fittingSize().width.max(LABEL_FRAME_MIN_W);
        let label_y = BOTTOM_PADDING + 7.0;
        let slider_y = BOTTOM_PADDING + 4.0;
        let elapsed_x = BOTTOM_HORIZONTAL_PADDING;
        let slider_x = elapsed_x + elapsed_w - 2.0;
        let duration_x = bw - BOTTOM_HORIZONTAL_PADDING - duration_w;
        let slider_w = (duration_x + 2.0 - slider_x).max(0.0);
        self.elapsed
            .setFrame(rect(elapsed_x, label_y, elapsed_w, LABEL_H));
        self.seek
            .setFrame(rect(slider_x, slider_y, slider_w, PLAY_SLIDER_FRAME_H));
        self.duration
            .setFrame(rect(duration_x, label_y, duration_w, LABEL_H));

        let frac = if st.duration > 0.0 {
            (st.position / st.duration).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // While AppKit is tracking a native slider, its value is the source of
        // truth. Writing playback state back into either slider during the same
        // mouse gesture can move the knob underneath AppKit's tracking loop and
        // produce a one-frame snap/flicker. Reconcile immediately after mouse-up.
        if !primary_mouse_down {
            self.seek.setDoubleValue(frac);
            self.volume.setDoubleValue(st.volume.clamp(0.0, 1.0) as f64);
        }

        if self.last_paused.get() != Some(st.paused) {
            self.last_paused.set(Some(st.paused));
            let icon = if st.paused {
                TransportIcon::Play
            } else {
                TransportIcon::Pause
            };
            self.play.setImage(Some(&transport_image(self.mtm, icon)));
        }

        // IINA converts the player's volume to Int before selecting the speaker
        // symbol, so 33.9% still uses wave.1 and 66.9% still uses wave.2.
        let volume_level = (st.volume.clamp(0.0, 1.0) as f64 * 100.0).floor();
        if self.last_muted.get() != Some(st.muted)
            || self.last_volume_level.get() != Some(volume_level)
        {
            self.last_muted.set(Some(st.muted));
            self.last_volume_level.set(Some(volume_level));
            let name = if st.muted {
                "speaker.slash.fill"
            } else if volume_level <= 0.0 {
                "speaker.fill"
            } else if volume_level <= 33.0 {
                "speaker.wave.1.fill"
            } else if volume_level <= 66.0 {
                "speaker.wave.2.fill"
            } else {
                "speaker.wave.3.fill"
            };
            if let Some(img) = configured_symbol(name, 13.0, unsafe { NSFontWeightRegular }) {
                self.speaker_image.setImage(Some(&img));
            }
        }

        let _ = self.mtm;
    }
}
