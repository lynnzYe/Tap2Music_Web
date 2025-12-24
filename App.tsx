import React, { useState, useEffect, useCallback, useRef } from "react";
import Keyboard from "./components/Keyboard";
import Visualizer from "./components/Visualizer";
import { audioEngine } from "./src/services/AudioEngine";
import { midiManager } from "./src/services/MidiManager";
import {
  BaseInferenceEngine,
  engineMap,
  InferenceConfig,
  InferenceFactory,
  InferenceSubMode,
} from "./src/services/InferenceEngine";
import {
  NoteEvent,
  NOTE_COLORS,
  PIANO_CONFIG,
  FREEPLAY_KEY_MAP,
  TAP2MUSIC_KEY_MAP,
} from "./src/types/types";

import {
  Volume2,
  VolumeX,
  Trash2,
  Settings,
  Piano as PianoIcon,
  Sparkles,
  Zap,
  Sliders,
} from "lucide-react";
import "./App.css";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";

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
  const [subMode, setSubMode] = useState<InferenceSubMode>("uc");

  const [showInferenceParams, setShowInferenceParams] = useState(false);
  // Inference Config State
  const [infConfig, setInfConfig] = useState<InferenceConfig>({
    samplingType: "temperature",
    temperature: 0.8,
    topP: 0.85,
  });

  const [tapStatus, setTapStatus] = useState("");
  const [loadingModel, setLoadingModel] = useState(false);

  const engineRef = React.useRef<BaseInferenceEngine<any> | null>(null);
  const pressedComputerKeys = useRef<Set<string>>(new Set());

  useEffect(() => {
    async function switchTapModel() {
      // Release old engine
      engineRef.current?.dispose();
      if (mode === "tap2music") {
        const newEngine = InferenceFactory.create(subMode);
        engineRef.current = newEngine;
        setLoadingModel(true);
        try {
          setTapStatus(`Loading model: ${subMode}`);
          // Run lazy self-test
          await newEngine.selfTest();
          await newEngine.load();
          setTapStatus("Ready!");
        } catch (e) {
          console.error(e);
          toast.error("UC model self-test failed");
          setTapStatus("Model Test failed!");
        }
        setLoadingModel(false);
      } else {
        engineRef.current = null;
        setLoadingModel(false);
      }
    }

    switchTapModel();
  }, [mode, subMode]);

  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.updateConfig(infConfig);
    }
  }, [infConfig]);

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
      // console.debug("noteon: ", pitch, velocity);
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
      // Only trigger if inputPitch is not already active
      if (pitchMap.current.has(inputPitch)) return;

      let triggeredPitch = inputPitch;

      if (mode === "tap2music" && engineRef.current) {
        // Handle Predict by mode. Remember to prepare input
        const input = {
          pitch: inputPitch,
          now: performance.now(),
          velocity: velocity,
          chord: null,
        };
        triggeredPitch = engineRef.current.run(input);
      }

      // Save mapping and trigger note
      pitchMap.current.set(inputPitch, triggeredPitch);
      triggerNoteOn(triggeredPitch, velocity);
    },
    [mode, triggerNoteOn]
  );

  const noteOff = useCallback(
    (inputPitch: number) => {
      const triggeredPitch = pitchMap.current.get(inputPitch);
      if (triggeredPitch === undefined) return; // ignore unmapped note

      if (mode === "tap2music" && engineRef.current) {
        engineRef.current.updateNoteoff(performance.now());
      }

      triggerNoteOff(triggeredPitch);
      pitchMap.current.delete(inputPitch);
    },
    [mode, triggerNoteOff]
  );

  const noteOnRef = useRef(noteOn);
  const noteOffRef = useRef(noteOff);
  useEffect(() => {
    noteOnRef.current = noteOn;
    noteOffRef.current = noteOff;
  }, [noteOn, noteOff]);

  const clear = useCallback(() => {
    setActiveNotes(new Set());
    if (engineRef.current) engineRef.current.reset();
    noteEventsRef.current = [];
    setNoteEvents([]);
    pitchMap.current.clear();
    audioEngine.clear();
  }, []);

  const toggleMode = () => {
    setMode((prev) => (prev === "freeplay" ? "tap2music" : "freeplay"));
  };

  // Computer Keyboard Handlers
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.repeat) return;
      const key = e.key.toLowerCase();

      let pitch: number | undefined;
      if (mode === "freeplay") {
        pitch = FREEPLAY_KEY_MAP[key];
      } else {
        pitch = TAP2MUSIC_KEY_MAP[key];
      }

      if (pitch !== undefined) {
        pressedComputerKeys.current.add(key);
        noteOn(pitch, 100);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (pressedComputerKeys.current.has(key)) {
        pressedComputerKeys.current.delete(key);
        let pitch: number | undefined;
        if (mode === "freeplay") {
          pitch = FREEPLAY_KEY_MAP[key];
        } else {
          pitch = TAP2MUSIC_KEY_MAP[key];
        }
        if (pitch !== undefined) {
          noteOff(pitch);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [mode, noteOn, noteOff]);

  useEffect(() => {
    const initMidi = async () => {
      // Only initialize if not already done (optional but safer)
      await midiManager.init();
      setInputs(midiManager.getInputs());
      setOutputs(midiManager.getOutputs());

      // This adds a listener. By having [] as dependencies,
      // we ensure this code ONLY runs once.
      midiManager.onMessage((status, d1, d2) => {
        const type = status & 0xf0;
        if (type === 0x90 && d2 > 0) {
          noteOnRef.current(d1, d2);
        } else if (type === 0x80 || (type === 0x90 && d2 === 0)) {
          noteOffRef.current(d1);
        }
      });
    };
    initMidi();

    // Empty dependency array is critical here!
  }, []);

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

  const availableSubModes = (
    Object.entries(engineMap) as [
      InferenceSubMode,
      (typeof engineMap)[InferenceSubMode]
    ][]
  ).filter(([, v]) => v.factory !== null);

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-950 text-white overflow-hidden select-none">
      {/* Header Controls */}
      <header className="flex flex-col bg-slate-900/60 backdrop-blur-xl border-b border-white/5 z-50">
        <div className="flex items-center justify-between px-6 py-3">
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
        </div>
        {/* AI Sub-modes Bar (Only visible in Tap2Music mode) */}
        {mode === "tap2music" && (
          <div className="px-6 py-2 bg-indigo-950/20 border-t border-white/5 flex items-center justify-between animate-in fade-in duration-300">
            <div className="flex items-center gap-4">
              <span className="text-[9px] font-black text-indigo-400 uppercase tracking-widest">
                Inference Engine:
              </span>

              <div className="flex gap-1">
                {availableSubModes.map(([key, cfg]) => {
                  const Icon = cfg.icon;
                  return (
                    <button
                      key={key}
                      onClick={() => setSubMode(key)}
                      className={`flex items-center gap-2 px-3 py-1 rounded-full text-[10px] font-bold transition-all ${
                        subMode === key
                          ? "bg-indigo-600 text-white"
                          : "text-slate-400 hover:bg-white/5"
                      }`}
                    >
                      <Icon className="w-3 h-3" />
                      {cfg.label}
                    </button>
                  );
                })}

                <div className="w-[1px] h-4 bg-white/10 mx-2" />

                <button className="flex items-center gap-2 px-3 py-1 rounded-full text-[10px] font-bold text-slate-600 cursor-not-allowed italic">
                  More to come...
                </button>
              </div>
            </div>
            <button
              onClick={() => {
                setShowInferenceParams(!showInferenceParams);
                setShowSettings(false);
              }}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-[11px] uppercase font-black tracking-tighter transition-all ${
                showInferenceParams
                  ? "bg-indigo-500 text-white"
                  : "bg-indigo-500/10 text-indigo-300 hover:bg-indigo-500/20"
              }`}
            >
              <Sliders className="w-3.5 h-3.5" /> Engine Parameters
            </button>
          </div>
        )}
      </header>

      {/* Main View Area */}
      <main className="flex-grow flex flex-col relative overflow-hidden">
        <Visualizer noteEvents={noteEvents} activeNotes={activeNotes} />

        {/* Piano Component - Reduced Height */}
        <Keyboard
          activeNotes={activeNotes}
          onNoteOn={noteOn}
          onNoteOff={noteOff}
          showLabels={mode === "freeplay"}
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
        {/* Inference Parameter Overlay */}
        {showInferenceParams && mode === "tap2music" && (
          <div className="absolute top-4 right-6 w-80 bg-indigo-950/40 backdrop-blur-2xl border border-indigo-500/20 rounded-2xl p-6 shadow-2xl z-50 animate-in fade-in zoom-in-95 duration-200">
            <h2 className="text-xs font-bold uppercase tracking-widest text-indigo-400 mb-6 flex items-center gap-2">
              <Zap className="w-3.5 h-3.5" /> {subMode.toUpperCase()} Parameters
            </h2>

            <div className="space-y-6">
              <div>
                <label className="block text-[10px] font-bold text-indigo-300 mb-3 uppercase tracking-tighter">
                  Sampling Strategy
                </label>
                <div className="grid grid-cols-2 gap-2 bg-black/20 p-1 rounded-xl">
                  <button
                    onClick={() =>
                      setInfConfig((prev) => ({
                        ...prev,
                        samplingType: "temperature",
                      }))
                    }
                    className={`py-1.5 rounded-lg text-[10px] font-bold transition-all ${
                      infConfig.samplingType === "temperature"
                        ? "bg-indigo-600 text-white"
                        : "text-slate-500 hover:text-slate-300"
                    }`}
                  >
                    Temperature
                  </button>
                  <button
                    onClick={() =>
                      setInfConfig((prev) => ({
                        ...prev,
                        samplingType: "nucleus",
                      }))
                    }
                    className={`py-1.5 rounded-lg text-[10px] font-bold transition-all ${
                      infConfig.samplingType === "nucleus"
                        ? "bg-indigo-600 text-white"
                        : "text-slate-500 hover:text-slate-300"
                    }`}
                  >
                    Nucleus
                  </button>
                </div>
              </div>

              {infConfig.samplingType === "temperature" ? (
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-[10px] font-bold text-indigo-300 uppercase tracking-tighter">
                      Temperature
                    </label>
                    <span className="text-xs font-mono text-indigo-400">
                      {infConfig.temperature.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.05"
                    value={infConfig.temperature}
                    onChange={(e) =>
                      setInfConfig((prev) => ({
                        ...prev,
                        temperature: parseFloat(e.target.value),
                      }))
                    }
                    className="w-full h-1.5 bg-indigo-950 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                  />
                  <div className="flex justify-between mt-1">
                    <span className="text-[8px] text-slate-600">PRECISE</span>
                    <span className="text-[8px] text-slate-600">CREATIVE</span>
                  </div>
                </div>
              ) : (
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-[10px] font-bold text-indigo-300 uppercase tracking-tighter">
                      Nucleus Sampling (Top-P)
                    </label>
                    <span className="text-xs font-mono text-indigo-400">
                      {infConfig.topP.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={infConfig.topP}
                    onChange={(e) =>
                      setInfConfig((prev) => ({
                        ...prev,
                        topP: parseFloat(e.target.value),
                      }))
                    }
                    className="w-full h-1.5 bg-indigo-950 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                  />
                  <div className="flex justify-between mt-1">
                    <span className="text-[8px] text-slate-600">NARROW</span>
                    <span className="text-[8px] text-slate-600">BROAD</span>
                  </div>
                </div>
              )}

              <div className="pt-4 border-t border-white/5">
                <p className="text-[10px] text-indigo-300/60 leading-relaxed italic">
                  These parameters control the randomness and focus of the model
                  output during the inference phase.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Loading overlay */}
      {loadingModel && (
        <div className="absolute inset-0 flex flex-col justify-center items-center  z-50">
          <div className="spinner"></div>
          <p>{tapStatus}</p>
        </div>
      )}

      {/* toast */}
      <Toaster position="top-right" richColors duration={5000} />

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
          {inputs.length > 0 ? "MIDI AVAILABLE" : "MIDI DISCONNECTED"}
        </div>
      </div>
    </div>
  );
};

export default App;
