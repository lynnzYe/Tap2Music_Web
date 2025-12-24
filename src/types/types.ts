
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

// Computer Keyboard Mapping for Freeplay mode
export const FREEPLAY_KEY_MAP: Record<string, number> = {
  // Center Row (E3 - C5)
  'q': 52, 'w': 53, 'e': 55, 'r': 57, 't': 59, 'y': 60, 'u': 62, 'i': 64, 'o': 65, 'p': 67, '[': 69, ']': 71, '\\': 72,
  '3': 54, '4': 56, '5': 58, '7': 61, '8': 63, '0': 66, '-': 68, '=': 70,
  // Lower Row (A2 - D3)
  'z': 45, 'x': 47, 'c': 48, 'v': 50,
  's': 46, 'f': 49, 'g': 51,
  // Upper Row (D5 - A5)
  'n': 74, 'm': 76, ',': 77, '.': 79, '/': 81,
  'j': 75, 'l': 78, ';': 80,
};

// Linear Mapping for Tap2Music mode
// We split the keyboard into 4 rows, each spanning the 21-108 range.
const createLinearMap = (keys: string[]) => {
  const map: Record<string, number> = {};
  keys.forEach((key, i) => {
    const ratio = i / (keys.length - 1);
    map[key] = Math.round(PIANO_CONFIG.MIN_MIDI + ratio * (PIANO_CONFIG.MAX_MIDI - PIANO_CONFIG.MIN_MIDI));
  });
  return map;
};

export const TAP2MUSIC_KEY_MAP: Record<string, number> = {
  ...createLinearMap(['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=']),
  ...createLinearMap(['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\']),
  ...createLinearMap(['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\'']),
  ...createLinearMap(['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/']),
};

// Map MIDI pitch to its white key index (0-51) or null if it's black
export const getWhiteKeyIndex = (midi: number): number | null => {
  const noteInOctave = midi % 12;
  const isBlack = [1, 3, 6, 8, 10].includes(noteInOctave);
  if (isBlack) return null;

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
