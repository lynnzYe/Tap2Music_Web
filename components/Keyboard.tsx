import React, { useState, useEffect, useRef } from "react";
import {
  PIANO_CONFIG,
  getKeyLayout,
  FREEPLAY_KEY_MAP,
} from "../src/types/types";

interface KeyboardProps {
  activeNotes: Set<number>;
  onNoteOn: (pitch: number, velocity: number, isGtInject?: boolean) => void;
  onNoteOff: (pitch: number) => void;
  showLabels?: boolean;
}

const Keyboard: React.FC<KeyboardProps> = ({
  activeNotes,
  onNoteOn,
  onNoteOff,
  showLabels,
}) => {
  const [showHint, setShowHint] = useState(false);
  const hintTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Invert the FREEPLAY_KEY_MAP to find labels for a given pitch
  const pitchToLabel = React.useMemo(() => {
    const map: Record<number, string> = {};
    Object.entries(FREEPLAY_KEY_MAP).forEach(([key, pitch]) => {
      map[pitch] = key;
    });
    return map;
  }, []);

  const isBlackNote = (midi: number) => {
    const note = midi % 12;
    return [1, 3, 6, 8, 10].includes(note);
  };

  // Trigger hint only when labels are turned OFF
  useEffect(() => {
    if (showLabels === false) {
      setShowHint(true);

      if (hintTimeoutRef.current) clearTimeout(hintTimeoutRef.current);

      hintTimeoutRef.current = setTimeout(() => {
        setShowHint(false);
      }, 3000);
    } else {
      // Hide immediately if labels are turned back on
      setShowHint(false);
    }
  }, [showLabels]);

  const whiteKeys = [];
  const blackKeys = [];

  for (let i = PIANO_CONFIG.MIN_MIDI; i <= PIANO_CONFIG.MAX_MIDI; i++) {
    const isBlack = isBlackNote(i);
    const active = activeNotes.has(i);
    const layout = getKeyLayout(i);
    const label = showLabels ? pitchToLabel[i] : null;

    const keyElement = (
      <div
        key={i}
        className={`group absolute select-none cursor-pointer flex flex-col items-center justify-end
          ${
            active
              ? isBlack
                ? "bg-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.6)]"
                : "bg-indigo-100 shadow-[inset_0_-4px_0_rgba(99,102,241,0.5)]"
              : isBlack
                ? "bg-neutral-900"
                : "bg-white"
          }
          transition-colors duration-75 border-b-2 border-slate-300
        `}
        style={{
          left: `${layout.x}%`,
          width: `${layout.width}%`,
          height: isBlack ? "60%" : "100%",
          zIndex: isBlack ? 20 : 10,
          borderRadius: isBlack ? "0 0 3px 3px" : "0 0 5px 5px",
          borderRight: !isBlack ? "1px solid #e2e8f0" : "none",
        }}
        onMouseDown={(e) => {
          e.preventDefault();
          onNoteOn(i, 100, true);
        }}
        onMouseUp={() => onNoteOff(i)}
        onMouseEnter={(e) => {
          if (e.buttons === 1) onNoteOn(i, 100, true);
        }}
        onMouseLeave={(e) => e.buttons === 1 && onNoteOff(i)}
      >
        {label && (
          <span
            className={`absolute ${isBlack ? "bottom-2" : "bottom-8"} left-1/2 -translate-x-1/2 font-black text-[9px] uppercase pointer-events-none ${isBlack ? "text-indigo-400" : "text-slate-500"}`}
          >
            {label}
          </span>
        )}

        {/* Subtle GT hint on hover when labels are off */}
        {!showLabels && (
          <span className="absolute bottom-6 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none text-[7px] font-black px-1 rounded-sm bg-indigo-600 text-white z-30">
            GT
          </span>
        )}

        {!isBlack && (i % 12 === 0 || i === 21) && (
          <span className="mb-1 text-[8px] text-slate-400 font-bold pointer-events-none">
            {i % 12 === 0 ? `C${Math.floor(i / 12) - 1}` : "A0"}
          </span>
        )}
      </div>
    );

    if (isBlack) blackKeys.push(keyElement);
    else whiteKeys.push(keyElement);
  }

  return (
    <div className="w-full h-32 bg-slate-950 border-t-2 border-slate-800 relative shadow-2xl overflow-hidden shrink-0">
      {/* One-time Onboarding Hint */}
      <div
        className={`absolute top-3 left-1/2 -translate-x-1/2 z-50 pointer-events-none transition-all duration-700
          ${showHint ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-4"}`}
      >
        <div className="flex items-center gap-2 bg-indigo-600/90 backdrop-blur-sm px-4 py-1 rounded-full border border-indigo-400 shadow-xl">
          <span className="text-[10px] text-white font-bold tracking-[0.2em] whitespace-nowrap">
            Keyboard Clicks Inject Ground Truth
          </span>
        </div>
      </div>

      <div className="relative w-full h-full">
        {whiteKeys}
        {blackKeys}
      </div>
    </div>
  );
};

export default Keyboard;
