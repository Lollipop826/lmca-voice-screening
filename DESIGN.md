---
name: Clinical Voice Screening Console
description: A calm, staff-first clinical interface for voice-based cognitive screening.
colors:
  primary: "#438EA0"
  primary-dark: "#2C6E7F"
  primary-legacy: "#4A90A4"
  accent-care: "#E8925B"
  background: "#F3F7F9"
  background-soft: "#F5F7FA"
  surface: "#FFFFFF"
  surface-wash: "#F7FAFB"
  surface-soft: "#EAF4F4"
  surface-control: "#EDF3F8"
  line: "#D9E3E8"
  line-soft: "#E2E8F0"
  ink: "#243142"
  ink-strong: "#111827"
  muted: "#69798F"
  muted-legacy: "#718096"
  status-listening: "#D69E2E"
  status-thinking: "#4D617C"
  status-speaking: "#38A169"
  status-speaking-deep: "#23724F"
  status-error: "#F56565"
  status-info: "#2459C8"
typography:
  display:
    fontFamily: '"Noto Sans SC", "Nunito", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "1.25rem"
    fontWeight: 800
    lineHeight: 1.2
    letterSpacing: "0"
  title:
    fontFamily: '"Noto Sans SC", "Nunito", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "1rem"
    fontWeight: 800
    lineHeight: 1.25
    letterSpacing: "0"
  body:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.68
    letterSpacing: "0"
  label:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "0.8rem"
    fontWeight: 760
    lineHeight: 1.35
    letterSpacing: "0"
rounded:
  xs: "6px"
  sm: "8px"
  md: "12px"
  lg: "14px"
  xl: "16px"
  panel: "18px"
  control: "28px"
  pill: "999px"
spacing:
  xxs: "4px"
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "20px"
  xl: "24px"
  xxl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface}"
    rounded: "{rounded.md}"
    padding: "0.8rem 1rem"
    height: "48px"
  button-secondary:
    backgroundColor: "{colors.surface-control}"
    textColor: "{colors.muted}"
    rounded: "{rounded.md}"
    height: "48px"
    width: "48px"
  button-danger:
    backgroundColor: "{colors.status-error}"
    textColor: "{colors.surface}"
    rounded: "{rounded.md}"
    height: "48px"
  input-field:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink-strong}"
    rounded: "{rounded.md}"
    padding: "0.8rem"
    height: "46px"
  settings-card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.lg}"
    padding: "16px"
  status-badge:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.panel}"
    padding: "10px 18px"
  chip-selected:
    backgroundColor: "{colors.surface-soft}"
    textColor: "{colors.primary-dark}"
    rounded: "{rounded.md}"
    padding: "10px 12px"
  control-bar:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "14px 22px"
---

# Design System: Clinical Voice Screening Console

## 1. Overview

**Creative North Star: "The Calm Screening Console"**

This system should feel like a quiet clinical instrument held by a nurse or screening staff member during a live conversation. It is warm enough to lower pressure, but disciplined enough that the operator can trust every state, control, and recovery path without rereading the screen.

The interface prioritizes glanceable state over decoration. Listening, thinking, speaking, recording, paused, scoring, and recovery states must be readable from shape, label, placement, and motion, not color alone. The patient should never feel tested by the screen; the staff member should feel quietly supported.

The visual language rejects marketing-site styling, decorative spectacle, loud technology aesthetics, dense hospital-admin clutter, childlike wellness visuals, and any layout that exposes too much operational complexity during a live conversation.

**Key Characteristics:**
- Restrained clinical color with one active blue-green action color.
- Large, reachable controls for one-handed phone use.
- A single state center: the clinical status orb plus text badge.
- Flat-by-default panels with small, structural shadows only where depth helps.
- Short state copy, clear recovery paths, and no decorative motion.

**The One Glance Rule.** A staff member should know the session state within one glance, even under poor clinical lighting and while facing the patient.

## 2. Colors

The palette is a cool clinical neutral base with a restrained blue-green action color and warm status accents reserved for meaning.

### Primary
- **Clinical Blue-Green** (`primary`): Use for primary actions, current selection, status icons, active links, and the microphone action. It should be visible but not decorative.
- **Deep Clinical Teal** (`primary-dark`): Use for brand marks, hover text, strong selected states, and quiet emphasis.
- **Legacy Medical Teal** (`primary-legacy`): Preserve for compatibility with older page fragments. New UI should use `primary` unless an existing fragment already depends on the legacy tone.

### Secondary
- **Care Orange** (`accent-care`): Use sparingly for warmth, patient-aware hints, and secondary status detail. It must not become a marketing accent.

### Tertiary
- **Listening Gold** (`status-listening`): Use only for listening or attention-waiting states.
- **Thinking Steel** (`status-thinking`): Use for processing, scoring, loading, and agent reasoning states.
- **Speaking Green** (`status-speaking`, `status-speaking-deep`): Use for assistant speech, active voice output, and confirmation states.
- **Clinical Error Red** (`status-error`): Use for required missing fields, destructive action warnings, and critical recording failures.
- **Information Blue** (`status-info`): Use for mode-active states and neutral informational selection.

### Neutral
- **Clinical Background** (`background`, `background-soft`): Use for page body and broad app surfaces.
- **Clean Surface** (`surface`): Use for cards, drawers, message bubbles, form fields, and sticky control panels.
- **Drawer Wash** (`surface-wash`): Use for right-side configuration panels and quiet secondary surfaces.
- **Selected Surface** (`surface-soft`): Use for selected chips, admin shortcuts, and active field groups.
- **Control Surface** (`surface-control`): Use for secondary icon buttons and inactive controls.
- **Clinical Line** (`line`, `line-soft`): Use for borders, dividers, and input strokes.
- **Ink** (`ink`, `ink-strong`): Use for primary text, headings, and high-confidence labels.
- **Muted Ink** (`muted`, `muted-legacy`): Use for helper text and placeholder text only when contrast remains clear.

**The Less Than Ten Rule.** The primary color should cover less than ten percent of any normal screen. It carries action and state, not decoration.

**The State Color Rule.** Gold, steel, green, red, and info blue are reserved for semantic state. Never use them as random accent colors.

## 3. Typography

**Display Font:** Noto Sans SC with Nunito and Chinese system fallbacks.
**Body Font:** Apple/system UI stack with Noto Sans SC, PingFang SC, and Microsoft YaHei fallbacks.
**Label/Mono Font:** Use the body stack for UI labels. Reserve mono only for debug or event-log tooling.

**Character:** The typography is utilitarian, warm, and compact. It should read like clinical software, not a landing page, and it should stay legible on phone-sized screens.

### Hierarchy
- **Display** (800, 1.25rem, 1.2): Use for drawer titles, main section titles, and compact page-level identity. Product screens do not use oversized display type.
- **Title** (800, 1rem, 1.25): Use for settings section headings, card titles, and summary panels.
- **Body** (400 to 600, 1rem, 1.68): Use for message bubbles and readable clinical text. Keep long prose near 65 to 75 characters when it is not chat content.
- **Label** (760, 0.8rem, 1.35): Use for form labels, section captions, compact badges, and helper labels. Avoid all-caps unless the label is very short and operational.
- **Micro** (650 to 700, 0.66rem to 0.78rem): Use for status chips, score badges, and tiny contextual hints.

**The No Hero Type Rule.** This is a task interface. Do not introduce oversized marketing headings, dramatic type pairings, or display fonts in controls.

**The Chinese First Rule.** Chinese UI copy must stay comfortable at 16px inputs and around 1rem message text. Never shrink key clinical labels below 0.8rem.

## 4. Elevation

The system uses a hybrid of tonal layering and restrained shadows. Most panels are flat at rest. Shadows appear on floating controls, conversation bubbles, the status orb, the drawer, and temporary overlays where depth clarifies stacking.

### Shadow Vocabulary
- **Ambient Low** (`0 2px 7px rgba(30, 55, 64, 0.05)`): Use for small top-bar chips and low-emphasis utility controls.
- **Message AI** (`0 3px 8px rgba(15, 23, 42, 0.04)`): Use for assistant message bubbles.
- **Message User** (`0 8px 16px rgba(44, 110, 127, 0.16)`): Use for user message bubbles where the active teal surface needs depth.
- **Panel Medium** (`0 8px 18px rgba(15, 23, 42, 0.05)`): Use for memory panels and stable floating cards.
- **Control Bar** (`0 16px 34px rgba(28, 55, 66, 0.12)`): Use only for the bottom fixed action bar.
- **Drawer Edge** (`-12px 0 28px rgba(31, 50, 58, 0.14)`): Use only for right-side drawers and configuration panels.
- **Orb Clinical** (`0 22px 44px rgba(44, 110, 127, 0.22)`): Use only for the desktop status orb.
- **Focus Ring** (`0 0 0 3px rgba(67, 142, 160, 0.12)`): Use for focused inputs and selected interactive controls.

**The Flat Until Useful Rule.** A card with a border usually gets no shadow. A shadow must explain position, focus, or interaction.

**The Drawer Edge Rule.** The right-side drawer uses a single edge shadow and border. Do not add nested shadowed cards inside it.

## 5. Components

### Buttons
- **Shape:** Gently squared clinical controls (`12px` for primary buttons, `14px` for desktop icon buttons, `999px` only for true pills).
- **Primary:** Use Clinical Blue-Green on white text with a minimum height of `46px` on mobile and `48px` on desktop.
- **Hover / Focus:** Hover may darken to Deep Clinical Teal. Focus uses the teal focus ring. Active states may compress subtly, but never bounce.
- **Secondary / Ghost:** Use Control Surface with muted text for inactive icon buttons. Selected mode uses a blue info wash and a clear focus ring.
- **Danger:** Use Clinical Error Red only for destructive or recording failure states.

### Clinical Status Orb
- **Character:** The orb is the signature state instrument, not decoration.
- **Desktop:** Use a circular blue-green core with an inner highlight, slow breathing, and a thin conic ring. The center label uses one Chinese character for the current state.
- **Mobile:** Collapse to a compact status dot inside a small rounded tile. The text badge carries most of the information.
- **States:** Idle uses blue-green, listening uses gold, thinking uses steel, speaking uses green.
- **Reduced Motion:** Disable ring drift, breath, and pulse when reduced motion is requested.

### Status Badge
- **Shape:** Rounded panel (`18px`) with white surface, subtle line, and compact icon tile.
- **Copy:** Use short operational Chinese copy such as "正在听", "正在思考", "正在说话", "已暂停".
- **Behavior:** Keep width constrained and truncate long state text rather than pushing controls.

### Conversation Bubbles
- **Assistant:** White surface, soft border, low shadow, rounded corners with a small tail notch.
- **User:** Blue-green gradient, white text, slightly stronger shadow, and a mirrored tail notch.
- **Typography:** Use body text with generous line height (`1.68`) so clinical prompts stay calm and readable.

### Drawer / Settings Panel
- **Style:** Right-side drawer on Drawer Wash, white internal sections, sticky header, sticky action footer.
- **Motion:** Enter quickly, around `160ms` to `180ms`, with an ease-out curve. Slow drawer movement is forbidden.
- **Density:** Use two-column fields where space allows, collapse to one column under narrow widths.
- **Sections:** Cards use `14px` radius, single border, no shadow. Shortcut links are `12px` to `13px` radius with clear hover feedback.

### Inputs / Fields
- **Style:** White field, Clinical Line stroke, `12px` radius, minimum height `44px` on mobile and `46px` on desktop.
- **Focus:** Border shifts to Clinical Blue-Green and uses the teal focus ring.
- **Placeholder:** Muted Ink must remain readable. Do not use faint gray.
- **Error:** Red border, soft red background, and a red focus ring. Include text, not color alone.

### Chips / Segmented Controls
- **Default:** White surface, Clinical Line border, Muted Ink.
- **Selected:** Selected Surface with Deep Clinical Teal text and a subtle inset teal line.
- **Tap Target:** Minimum height `44px` for education and mode options.

### Bottom Control Bar
- **Desktop:** Fixed centered panel, `28px` radius, white surface, thin line, and Control Bar shadow.
- **Mobile:** Occupies the safe-area bottom zone with large reachable actions.
- **Primary Action:** The microphone is the visual anchor. Secondary controls must not compete with it.

### Cards / Panels
- **Corner Style:** Use `14px` to `16px` for content panels. Avoid larger card radii.
- **Background:** White for content, Drawer Wash for settings background, Control Surface for inactive controls.
- **Border:** Use a single subtle clinical line. No colored side stripes.
- **Padding:** Use `14px` to `16px` for compact settings cards and `18px` to `22px` for larger panels.

## 6. Do's and Don'ts

### Do:
- **Do** optimize for staff first and patient protection. The patient participates through voice and should not need to read or operate the screen.
- **Do** make listening, speaking, thinking, recording, paused, scoring, and recovery states unmistakable through copy, placement, shape, and motion.
- **Do** keep primary actions large, reachable, and resilient on a phone during a live session.
- **Do** use gentle recovery language for refusals, uncertainty, long pauses, and interruptions.
- **Do** preserve functional fidelity while using this calmer mobile-first visual system.
- **Do** verify WCAG AA contrast for text, controls, placeholders, and status chips.
- **Do** include reduced-motion alternatives for every animated state.

### Don't:
- **Don't** copy the existing `static/voice_chat.html` visual UI as a template for new screens; use it only as a functional reference and apply this system.
- **Don't** use marketing-site styling.
- **Don't** use decorative spectacle.
- **Don't** use loud technology aesthetics.
- **Don't** create dense hospital-admin clutter.
- **Don't** use childlike wellness visuals.
- **Don't** create anything that makes the patient feel tested, rushed, or corrected.
- **Don't** expose too much operational complexity to the staff member during the live conversation.
- **Don't** use purple gradients, glass cards, neon effects, or decorative blur as the default product style.
- **Don't** add colored side stripes, oversized card radii, or border plus heavy shadow ghost cards.
- **Don't** animate the drawer slowly. Configuration panels must appear promptly and predictably.
