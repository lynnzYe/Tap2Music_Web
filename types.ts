
export interface NoteEvent {
  id: string;
  pitch: number;
  velocity: number;
  startTime: number; // in milliseconds
  endTime?: number;  // in milliseconds
  color: string;
}

export interface PianoSettings {
  isMuted: boolean;
  midiOutputEnabled: boolean;
  midiInputDevice: string | null;
  midiOutputDevice: string | null;
  volume: number;
}

export const PIANO_CONFIG = {
  MIN_MIDI: 21, // A0
  MAX_MIDI: 108, // C8
  NUM_KEYS: 88,
  NUM_WHITE_KEYS: 52,
  SCROLL_SPEED: 0.15, // pixels per ms
};

// Map MIDI pitch to its white key index (0-51) or null if it's black
export const getWhiteKeyIndex = (midi: number): number | null => {
  const noteInOctave = midi % 12;
  const octave = Math.floor(midi / 12) - 1;
  const isBlack = [1, 3, 6, 8, 10].includes(noteInOctave);
  if (isBlack) return null;

  // Offset mapping for C, D, E, F, G, A, B within an octave
  const whiteOffsets: Record<number, number> = { 0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6 };
  // Starting note is A0 (midi 21). Let's calculate from A0.
  // A is index 5 in whiteOffsets.
  // A0 is octave 0, note 9. 
  // Simplified: 
  let count = 0;
  for (let i = 21; i < midi; i++) {
    const n = i % 12;
    if (![1, 3, 6, 8, 10].includes(n)) count++;
  }
  return count;
};

// Returns { x: percentage, width: percentage } for any MIDI pitch
export const getKeyLayout = (midi: number) => {
  const whiteWidth = 100 / PIANO_CONFIG.NUM_WHITE_KEYS;
  const noteInOctave = midi % 12;
  const isBlack = [1, 3, 6, 8, 10].includes(noteInOctave);

  if (!isBlack) {
    const index = getWhiteKeyIndex(midi) ?? 0;
    return { x: index * whiteWidth, width: whiteWidth };
  } else {
    // Black keys are between white keys. 
    // Example: A#0 (22) is between A0 (21) and B0 (23).
    // It should be centered on the line between them.
    const prevWhiteIndex = getWhiteKeyIndex(midi - 1) ?? 0;
    const blackWidth = whiteWidth * 0.65;
    return { 
      x: (prevWhiteIndex + 1) * whiteWidth - (blackWidth / 2), 
      width: blackWidth 
    };
  }
};

export const NOTE_COLORS: Record<number, string> = {
  0: '#f43f5e', // C
  1: '#fb923c', // C#
  2: '#facc15', // D
  3: '#a3e635', // D#
  4: '#22c55e', // E
  5: '#2dd4bf', // F
  6: '#38bdf8', // F#
  7: '#60a5fa', // G
  8: '#818cf8', // G#
  9: '#a855f7', // A
  10: '#d946ef', // A#
  11: '#f472b6', // B
};
