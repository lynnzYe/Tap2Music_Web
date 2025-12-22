
import React from 'react';
import { PIANO_CONFIG, getKeyLayout } from '../types';

interface KeyboardProps {
  activeNotes: Set<number>;
  onNoteOn: (pitch: number, velocity: number) => void;
  onNoteOff: (pitch: number) => void;
}

const Keyboard: React.FC<KeyboardProps> = ({ activeNotes, onNoteOn, onNoteOff }) => {
  const isBlackNote = (midi: number) => {
    const note = midi % 12;
    return [1, 3, 6, 8, 10].includes(note);
  };

  const handleMouseDown = (midi: number) => {
    onNoteOn(midi, 100);
  };

  const handleMouseUp = (midi: number) => {
    onNoteOff(midi);
  };

  const handleMouseEnter = (e: React.MouseEvent, midi: number) => {
    if (e.buttons === 1) {
      onNoteOn(midi, 100);
    }
  };

  const handleMouseLeave = (e: React.MouseEvent, midi: number) => {
    if (e.buttons === 1) {
      onNoteOff(midi);
    }
  };

  const whiteKeys = [];
  const blackKeys = [];

  for (let i = PIANO_CONFIG.MIN_MIDI; i <= PIANO_CONFIG.MAX_MIDI; i++) {
    const isBlack = isBlackNote(i);
    const active = activeNotes.has(i);
    const layout = getKeyLayout(i);

    const keyElement = (
      <div
        key={i}
        className={`absolute select-none cursor-pointer flex flex-col items-center justify-end
          ${active 
            ? (isBlack ? 'bg-indigo-500 shadow-[inset_0_-2px_0_rgba(255,255,255,0.4)]' : 'bg-indigo-100 shadow-[inset_0_-4px_0_rgba(99,102,241,0.5)]') 
            : (isBlack ? 'bg-neutral-900 shadow-[inset_0_-3px_0_rgba(0,0,0,0.8)]' : 'bg-white shadow-[inset_0_-4px_0_rgba(0,0,0,0.1)]')}
          transition-colors duration-75 border-b-2 border-slate-300
        `}
        style={{
          left: `${layout.x}%`,
          width: `${layout.width}%`,
          height: isBlack ? '60%' : '100%',
          zIndex: isBlack ? 20 : 10,
          borderRadius: isBlack ? '0 0 3px 3px' : '0 0 5px 5px',
          borderRight: !isBlack ? '1px solid #e2e8f0' : 'none',
        }}
        onMouseDown={(e) => { e.preventDefault(); handleMouseDown(i); }}
        onMouseUp={() => handleMouseUp(i)}
        onMouseEnter={(e) => handleMouseEnter(e, i)}
        onMouseLeave={(e) => handleMouseLeave(e, i)}
        onContextMenu={(e) => e.preventDefault()}
      >
        {!isBlack && (i % 12 === 0 || i === 21) && (
          <span className="mb-2 text-[8px] text-slate-400 font-bold pointer-events-none uppercase tracking-tighter">
             {i % 12 === 0 ? `C${Math.floor(i / 12) - 1}` : 'A0'}
          </span>
        )}
      </div>
    );

    if (isBlack) blackKeys.push(keyElement);
    else whiteKeys.push(keyElement);
  }

  return (
    // Reduced height from h-44 to h-32
    <div className="w-full h-32 bg-slate-950 border-t-2 border-slate-800 relative shadow-2xl overflow-hidden shrink-0">
      <div className="relative w-full h-full">
        {whiteKeys}
        {blackKeys}
      </div>
    </div>
  );
};

export default Keyboard;
