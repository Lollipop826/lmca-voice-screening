# Product

## Register

product

## Users

The primary users are doctors, nurses, and cognitive screening staff using a phone-sized interface during an in-person screening session. They operate the controls, monitor the conversation state, and handle follow-up actions. Patients do not operate the UI directly; they participate through voice and should not need to read, tap, or understand the screen.

## Product Purpose

This product supports low-pressure voice-based cognitive screening in a clinical or care setting. It helps staff guide a patient through conversational MMSE-style tasks, capture voice interaction, manage interruptions or refusal, show task media when needed, and review scores or session history. Success means the staff member can keep attention on the patient while the interface quietly handles state, scoring, prompts, and recovery.

## Brand Personality

Professional, warm, low-pressure. The product should feel clinically reliable without feeling cold, patient-aware without becoming childish, and calm enough for repeated use in a screening room.

## Anti-references

Do not copy the existing `static/voice_chat.html` visual UI; use it only as a functional reference. Avoid marketing-site styling, decorative spectacle, loud technology aesthetics, dense hospital-admin clutter, childlike wellness visuals, and anything that makes the patient feel tested, rushed, or corrected. Avoid designs that expose too much operational complexity to the staff member during the live conversation.

## Design Principles

1. Staff first, patient protected: optimize for the clinician's workflow while keeping patient-facing interaction voice-only and low stress.
2. Calm state clarity: listening, speaking, thinking, recording, paused, scoring, and recovery states must be unmistakable at a glance.
3. One-handed clinical use: primary actions should be reachable, large, and resilient on a phone during a live session.
4. Gentle recovery: refusals, uncertainty, long pauses, and interruptions should feel handled, not punished.
5. Functional fidelity over visual inheritance: preserve the current page's capabilities, but redesign the mobile UI from a fresh visual system.

## Accessibility & Inclusion

Target WCAG AA contrast for all visible text and controls. Use large tap targets, readable type, clear focus states, and reduced-motion alternatives. Avoid relying on color alone for status. The interface should remain usable under clinical lighting, with older patients nearby, and with staff who may need to glance quickly rather than study the screen.
