
type MidiCallback = (status: number, data1: number, data2: number) => void;

export class MidiManager {
  private midiAccess: MIDIAccess | null = null;
  private callbacks: MidiCallback[] = [];
  private currentInput: MIDIInput | null = null;
  private currentOutput: MIDIOutput | null = null;

  async init() {
    if (!navigator.requestMIDIAccess) {
      console.warn("Web MIDI API not supported");
      return;
    }
    try {
      this.midiAccess = await navigator.requestMIDIAccess();
      this.midiAccess.onstatechange = (e) => {
        // Refresh lists or notify app
      };
    } catch (e) {
      console.error("MIDI access denied", e);
    }
  }

  getInputs() {
    return this.midiAccess ? Array.from(this.midiAccess.inputs.values()) : [];
  }

  getOutputs() {
    return this.midiAccess ? Array.from(this.midiAccess.outputs.values()) : [];
  }

  setInput(id: string | null) {
    if (this.currentInput) {
      this.currentInput.onmidimessage = null;
    }
    if (id && this.midiAccess) {
      const input = this.midiAccess.inputs.get(id);
      if (input) {
        this.currentInput = input;
        input.onmidimessage = (msg) => {
          const [status, d1, d2] = msg.data;
          this.callbacks.forEach(cb => cb(status, d1, d2));
        };
      }
    } else {
      this.currentInput = null;
    }
  }

  setOutput(id: string | null) {
    if (id && this.midiAccess) {
      this.currentOutput = this.midiAccess.outputs.get(id) || null;
    } else {
      this.currentOutput = null;
    }
  }

  send(status: number, d1: number, d2: number) {
    if (this.currentOutput) {
      this.currentOutput.send([status, d1, d2]);
    }
  }

  onMessage(cb: MidiCallback) {
    this.callbacks.push(cb);
  }
}

export const midiManager = new MidiManager();
