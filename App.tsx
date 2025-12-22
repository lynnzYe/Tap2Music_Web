import React, { useState, useEffect, useCallback, useRef } from "react";
import Keyboard from "./components/Keyboard";
import Visualizer from "./components/Visualizer";
import { audioEngine } from "./src/services/AudioEngine";
import { midiManager } from "./src/services/MidiManager";
import { inferenceEngine } from "./src/services/InferenceEngine";
import { NoteEvent, NOTE_COLORS, PIANO_CONFIG } from "./types";
import {
  Volume2,
  VolumeX,
  Trash2,
  Settings,
  Piano as PianoIcon,
  Sparkles,
  Zap,
} from "lucide-react";
import TapWrapper from "./src/model/TapWrapper";
import "./App.css";

type PlayMode = "freeplay" | "tap2music";

const App: React.FC = () => {
  const [activeNotes, setActiveNotes] = useState<Set<number>>(new Set());
  const [noteEvents, setNoteEvents] = useState<NoteEvent[]>([]);
  const [isMuted, setIsMuted] = useState(false);
  const [midiOutEnabled, setMidiOutEnabled] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [inputs, setInputs] = useState<any[]>([]);
  const [outputs, setOutputs] = useState<any[]>([]);
  const [mode, setMode] = useState<PlayMode>("freeplay");
  const [tapErr, setTapError] = useState(false);
  const [tapStatus, setTapStatus] = useState("Initializing model...");
  const [loadingModel, setLoadingModel] = useState(true);

  const tapRef = React.useRef<TapWrapper | null>(null);
  // Test Tap2Music Model
  useEffect(() => {
    async function initTap() {
      if (!window.my || !window.my.testUCTap || !window.my.TapConverter) {
        console.error("window or tap model undefined");
        setTapError(true);
        return;
      }
      setTapStatus("Running model self-test...");
      try {
        if (!window._midiTestRan) {
          window._midiTestRan = true;
          await window.my.testUCTap();
        }
        tapRef.current = new TapWrapper();
        await tapRef.current.load();
        setTapStatus("Ready!");
        setLoadingModel(false);
        setTapError(false);
      } catch (e) {
        console.error(e)
        setTapStatus("Model self-test failed! This should not happen.");
        setTapError(true);
        setLoadingModel(false);
      }
    }
    initTap();
  }, []);

  // Keep a ref to events for performance
  const noteEventsRef = useRef<NoteEvent[]>([]);
  // Maps incoming MIDI pitch to the pitch that was actually triggered (for Tap2Music mode)
  const pitchMap = useRef<Map<number, number>>(new Map());

  // Internal trigger for note effects
  const triggerNoteOn = useCallback(
    (pitch: number, velocity: number = 100) => {
      if (pitch < PIANO_CONFIG.MIN_MIDI || pitch > PIANO_CONFIG.MAX_MIDI)
        return;

      setActiveNotes((prev) => {
        const next = new Set(prev);
        next.add(pitch);
        return next;
      });

      audioEngine.noteOn(pitch, velocity);

      if (midiOutEnabled) {
        midiManager.send(0x90, pitch, velocity);
      }

      const newNote: NoteEvent = {
        id: Math.random().toString(36).substr(2, 9),
        pitch,
        velocity,
        startTime: performance.now(),
        color: NOTE_COLORS[pitch % 12],
      };

      noteEventsRef.current = [...noteEventsRef.current, newNote];
      setNoteEvents([...noteEventsRef.current]);
    },
    [midiOutEnabled]
  );

  const triggerNoteOff = useCallback(
    (pitch: number) => {
      setActiveNotes((prev) => {
        const next = new Set(prev);
        next.delete(pitch);
        return next;
      });

      audioEngine.noteOff(pitch);

      if (midiOutEnabled) {
        midiManager.send(0x80, pitch, 0);
      }

      const now = performance.now();
      noteEventsRef.current = noteEventsRef.current.map((n) =>
        n.pitch === pitch && !n.endTime ? { ...n, endTime: now } : n
      );
      setNoteEvents([...noteEventsRef.current]);
    },
    [midiOutEnabled]
  );

  // Public API exposed for the model/input
  const noteOn = useCallback(
    (inputPitch: number, velocity: number = 100) => {
      let triggeredPitch = inputPitch;

      if (mode === "tap2music") {
        triggeredPitch = inferenceEngine.predict(inputPitch, velocity);
      }

      pitchMap.current.set(inputPitch, triggeredPitch);
      triggerNoteOn(triggeredPitch, velocity);
    },
    [mode, triggerNoteOn]
  );

  const noteOff = useCallback(
    (inputPitch: number) => {
      const triggeredPitch = pitchMap.current.get(inputPitch) ?? inputPitch;
      triggerNoteOff(triggeredPitch);
      pitchMap.current.delete(inputPitch);
    },
    [triggerNoteOff]
  );

  const clear = useCallback(() => {
    setActiveNotes(new Set());
    noteEventsRef.current = [];
    setNoteEvents([]);
    pitchMap.current.clear();
    audioEngine.clear();
  }, []);

  const toggleMode = () => {
    setMode((prev) => (prev === "freeplay" ? "tap2music" : "freeplay"));
  };

  useEffect(() => {
    const initMidi = async () => {
      await midiManager.init();
      setInputs(midiManager.getInputs());
      setOutputs(midiManager.getOutputs());

      midiManager.onMessage((status, d1, d2) => {
        const type = status & 0xf0;
        if (type === 0x90 && d2 > 0) {
          noteOn(d1, d2);
        } else if (type === 0x80 || (type === 0x90 && d2 === 0)) {
          noteOff(d1);
        }
      });
    };
    initMidi();
  }, [noteOn, noteOff]);

  useEffect(() => {
    const interval = setInterval(() => {
      const now = performance.now();
      const cutoff = now - 10000;
      noteEventsRef.current = noteEventsRef.current.filter(
        (n) => (n.endTime || now) > cutoff
      );
      setNoteEvents([...noteEventsRef.current]);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loadingModel)
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
        <p>{tapStatus}</p>
      </div>
    );
  if (tapErr)
    return (
      <div className="error-screen">
        <h2>⚠️ Initialization failed</h2>
      </div>
    );

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-950 text-white overflow-hidden select-none">
      {/* Header Controls */}
      <header className="flex items-center justify-between px-6 py-3 bg-slate-900/60 backdrop-blur-xl border-b border-white/5 z-50">
        <div className="flex items-center gap-6">
          <div
            className="flex items-center gap-3 cursor-pointer group"
            onClick={toggleMode}
          >
            <div
              className={`p-2 rounded-lg transition-all duration-300 ${
                mode === "tap2music"
                  ? "bg-indigo-500 shadow-lg shadow-indigo-500/40 rotate-12"
                  : "bg-slate-700"
              }`}
            >
              <PianoIcon className="w-5 h-5 text-white" />
            </div>
            <div className="flex flex-col">
              <h1 className="text-xl font-black tracking-tighter leading-none flex items-center gap-2 group-hover:text-indigo-400 transition-colors">
                Tap2Music
                {mode === "tap2music" && (
                  <Sparkles className="w-4 h-4 text-indigo-400 animate-pulse" />
                )}
              </h1>
              <span
                className={`text-[9px] font-bold tracking-[0.2em] uppercase transition-colors ${
                  mode === "tap2music" ? "text-indigo-400" : "text-slate-500"
                }`}
              >
                {mode === "tap2music" ? "AI INFERENCE ON" : "FREEPLAY MODE"}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex bg-slate-800/50 p-1 rounded-xl border border-white/5 mr-2">
            <button
              onClick={() => setMode("freeplay")}
              className={`px-3 py-1.5 rounded-lg text-[10px] font-bold transition-all ${
                mode === "freeplay"
                  ? "bg-slate-700 text-white shadow-sm"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              FREEPLAY
            </button>
            <button
              onClick={() => setMode("tap2music")}
              className={`px-3 py-1.5 rounded-lg text-[10px] font-bold transition-all flex items-center gap-1.5 ${
                mode === "tap2music"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              <Zap className="w-3 h-3" /> TAP2MUSIC
            </button>
          </div>

          <button
            onClick={() => {
              setIsMuted(!isMuted);
              audioEngine.setMute(!isMuted);
            }}
            className={`p-2 rounded-xl transition-all ${
              isMuted
                ? "bg-red-500/20 text-red-500"
                : "bg-slate-800/50 text-slate-300 hover:bg-slate-700"
            }`}
            title={isMuted ? "Unmute" : "Mute"}
          >
            {isMuted ? (
              <VolumeX className="w-5 h-5" />
            ) : (
              <Volume2 className="w-5 h-5" />
            )}
          </button>

          <button
            onClick={clear}
            className="p-2 bg-slate-800/50 text-slate-300 hover:bg-slate-700 rounded-xl transition-all"
            title="Clear Visualizer"
          >
            <Trash2 className="w-5 h-5" />
          </button>

          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-2 rounded-xl transition-all ${
              showSettings
                ? "bg-indigo-600 text-white"
                : "bg-slate-800/50 text-slate-300 hover:bg-slate-700"
            }`}
            title="Settings"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Main View Area */}
      <main className="flex-grow flex flex-col relative overflow-hidden">
        <Visualizer noteEvents={noteEvents} activeNotes={activeNotes} />

        {/* Piano Component - Reduced Height */}
        <Keyboard
          activeNotes={activeNotes}
          onNoteOn={noteOn}
          onNoteOff={noteOff}
        />

        {/* Settings Overlay */}
        {showSettings && (
          <div className="absolute top-4 right-6 w-80 bg-slate-900/95 backdrop-blur-2xl border border-white/10 rounded-2xl p-6 shadow-2xl z-50 animate-in fade-in zoom-in-95 duration-200">
            <h2 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-6 flex items-center gap-2">
              <Settings className="w-3.5 h-3.5" /> MIDI Configuration
            </h2>

            <div className="space-y-6">
              <div>
                <label className="block text-[10px] font-bold text-slate-400 mb-2 uppercase tracking-tighter">
                  Input Device
                </label>
                <select
                  className="w-full bg-slate-800 border border-white/5 rounded-xl p-3 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all cursor-pointer"
                  onChange={(e) => midiManager.setInput(e.target.value)}
                >
                  <option value="">None / All</option>
                  {inputs.map((input) => (
                    <option key={input.id} value={input.id}>
                      {input.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-[10px] font-bold text-slate-400 mb-2 uppercase tracking-tighter">
                  Output Device
                </label>
                <select
                  className="w-full bg-slate-800 border border-white/5 rounded-xl p-3 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all cursor-pointer"
                  onChange={(e) => midiManager.setOutput(e.target.value)}
                >
                  <option value="">None</option>
                  {outputs.map((output) => (
                    <option key={output.id} value={output.id}>
                      {output.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-center justify-between py-2 border-t border-white/5">
                <span className="text-xs font-bold text-slate-300 uppercase">
                  Forward MIDI
                </span>
                <button
                  onClick={() => setMidiOutEnabled(!midiOutEnabled)}
                  className={`w-10 h-5 rounded-full p-1 transition-colors ${
                    midiOutEnabled ? "bg-indigo-600" : "bg-slate-700"
                  }`}
                >
                  <div
                    className={`w-3 h-3 bg-white rounded-full shadow-lg transition-transform ${
                      midiOutEnabled ? "translate-x-5" : "translate-x-0"
                    }`}
                  />
                </button>
              </div>

              <div className="pt-4 border-t border-white/5">
                <p className="text-[10px] text-slate-500 leading-relaxed font-medium">
                  Select IAC Driver to route output to external software.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Connection Indicator */}
      <div className="fixed bottom-36 left-6 z-50 pointer-events-none">
        <div className="flex items-center gap-2 bg-slate-900/60 backdrop-blur px-3 py-1.5 rounded-full border border-white/5 text-[9px] font-bold tracking-widest uppercase text-slate-400">
          <div
            className={`w-1.5 h-1.5 rounded-full ${
              inputs.length > 0
                ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]"
                : "bg-slate-600"
            }`}
          />
          {inputs.length > 0 ? "MIDI ACTIVE" : "MIDI DISCONNECTED"}
        </div>
      </div>
    </div>
  );
};

export default App;
