
export class AudioEngine {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private activeNotes: Map<number, { nodes: AudioNode[]; gain: GainNode }> = new Map();
  private muted: boolean = false;

  constructor() {}

  private init() {
    if (this.ctx) return;
    this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    
    // Compressor to prevent clipping during large chords
    const compressor = this.ctx.createDynamicsCompressor();
    compressor.threshold.setValueAtTime(-24, this.ctx.currentTime);
    compressor.knee.setValueAtTime(30, this.ctx.currentTime);
    compressor.ratio.setValueAtTime(12, this.ctx.currentTime);
    compressor.attack.setValueAtTime(0, this.ctx.currentTime);
    compressor.release.setValueAtTime(0.25, this.ctx.currentTime);

    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.setValueAtTime(0.5, this.ctx.currentTime);
    
    this.masterGain.connect(compressor);
    compressor.connect(this.ctx.destination);
  }

  public setMute(isMuted: boolean) {
    this.muted = isMuted;
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setTargetAtTime(isMuted ? 0 : 0.5, this.ctx.currentTime, 0.05);
    }
  }

  public noteOn(pitch: number, velocity: number) {
    this.init();
    if (!this.ctx || !this.masterGain || this.muted) return;

    if (this.activeNotes.has(pitch)) {
      this.noteOff(pitch);
    }

    const freq = Math.pow(2, (pitch - 69) / 12) * 440;
    const now = this.ctx.currentTime;
    const velFactor = velocity / 127;
    
    const nodes: AudioNode[] = [];
    const noteGain = this.ctx.createGain();
    noteGain.connect(this.masterGain);

    // Fundamental (Body)
    const osc1 = this.ctx.createOscillator();
    osc1.type = 'triangle';
    osc1.frequency.setValueAtTime(freq, now);
    const g1 = this.ctx.createGain();
    g1.gain.setValueAtTime(0.6 * velFactor, now);
    osc1.connect(g1);
    g1.connect(noteGain);
    osc1.start(now);
    nodes.push(osc1, g1);

    // Overtones (Brilliance)
    const osc2 = this.ctx.createOscillator();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(freq * 2, now);
    const g2 = this.ctx.createGain();
    g2.gain.setValueAtTime(0.2 * velFactor, now);
    osc2.connect(g2);
    g2.connect(noteGain);
    osc2.start(now);
    nodes.push(osc2, g2);

    // Hammer Strike Simulation (Percussive noise)
    const noise = this.ctx.createBufferSource();
    const bufferSize = this.ctx.sampleRate * 0.02; // 20ms burst
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) data[i] = Math.random() * 2 - 1;
    noise.buffer = buffer;

    const noiseFilter = this.ctx.createBiquadFilter();
    noiseFilter.type = 'highpass';
    noiseFilter.frequency.setValueAtTime(freq * 2, now);

    const noiseGain = this.ctx.createGain();
    noiseGain.gain.setValueAtTime(velFactor * 0.3, now);
    noiseGain.gain.exponentialRampToValueAtTime(0.01, now + 0.03);

    noise.connect(noiseFilter);
    noiseFilter.connect(noiseGain);
    noiseGain.connect(noteGain);
    noise.start(now);
    nodes.push(noise, noiseFilter, noiseGain);

    // Envelope
    noteGain.gain.setValueAtTime(0, now);
    noteGain.gain.linearRampToValueAtTime(1, now + 0.005);
    noteGain.gain.exponentialRampToValueAtTime(0.4, now + 0.1);
    noteGain.gain.exponentialRampToValueAtTime(0.01, now + 4.0);

    this.activeNotes.set(pitch, { nodes, gain: noteGain });
  }

  public noteOff(pitch: number) {
    if (!this.ctx) return;
    const active = this.activeNotes.get(pitch);
    if (!active) return;

    const now = this.ctx.currentTime;
    active.gain.gain.cancelScheduledValues(now);
    active.gain.gain.setValueAtTime(active.gain.gain.value, now);
    active.gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.1);

    const localNodes = active.nodes;
    const localGain = active.gain;

    setTimeout(() => {
      localNodes.forEach(n => {
        if (n instanceof OscillatorNode || n instanceof AudioBufferSourceNode) {
          try { n.stop(); } catch(e) {}
        }
        n.disconnect();
      });
      localGain.disconnect();
    }, 150);

    this.activeNotes.delete(pitch);
  }

  public clear() {
    this.activeNotes.forEach((_, pitch) => this.noteOff(pitch));
  }
}

export const audioEngine = new AudioEngine();
