
import { UserCircle, Hand, Layers } from "lucide-react";

type RawTapContext = {
  now: number;
  pitch?: number | null; // currently performed MIDI pitch, used for range/hand condition
  velocity?: number | null;
  chord?: number | null; // for chord-conditioned model, requires further logic to extract chords on the fly
};

type UCPredictInput = {
  time: number
  velocity?: number | null
};

type HandPredictInput = {
  time: number
  velocity?: number | null
  hand?: number | null
};

type DummyInput = {
  pitch?: number | null
}

type InferenceInputMap = {
  uc: UCPredictInput;
  hand: HandPredictInput;
  dummy: DummyInput;
  // experimental: ExperimentalPredictInput;
};

export type SamplingType = 'temperature' | 'nucleus';

export interface InferenceConfig {
  samplingType: SamplingType;
  temperature: number;
  topP: number;
}

const modelTestStatus: Record<InferenceSubMode, boolean> = {
  uc: false,
  hand: false,
  experimental: false,
  dummy: true
};

export abstract class BaseInferenceEngine<TInput> {
  public tapper: any;
  abstract readonly kind: keyof InferenceInputMap;
  protected config: InferenceConfig = {
    samplingType: 'temperature',
    temperature: 0.8,
    topP: 0.85,
  };

  // public API (uniform)
  run(ctx: RawTapContext): number {
    const input = this.prepareInput(ctx);
    return this.predict(input);
  }

  async load() { await this.tapper.init(); this.initInference() }

  reset() { this.tapper.reset(); this.initInference() }

  updateNoteoff(noteoffTime: number) { this.tapper.updateDur(noteoffTime) }

  dispose() {
    console.debug(`Disposing engine: ${this.constructor.name}`);
    this.tapper.dispose()
  }

  async selfTest(): Promise<void> {
    // Default: do nothing
  }

  public updateConfig(config: Partial<InferenceConfig>) {
    this.config = { ...this.config, ...config };
    console.debug(`Engine config updated:`, this.config);
  }

  protected predict(input: TInput): number {
    let inputWithConfig = { ...input, ...this.config }
    console.debug("predict with input:", inputWithConfig)
    return this.tapper.predict(inputWithConfig);
  }

  // Internel methods to be implemented
  protected abstract initInference(): void; // PAD to mark Start of New Sequence
  protected abstract prepareInput(ctx: RawTapContext): TInput;
}

class UCInferenceEngine extends BaseInferenceEngine<UCPredictInput> {
  readonly kind = 'uc';

  constructor() {
    super();
    if (!window.my?.UCTapWrapper) throw new Error("Not loaded");
    this.tapper = new window.my.UCTapWrapper();
  }

  async selfTest() {
    if (window.my?.testUCTap && !modelTestStatus.uc) {
      await window.my.testUCTap();
      modelTestStatus.uc = true;
      console.debug("UC model self-test passed");
    }
  }

  protected initInference() {
    this.predict(this.prepareInput({ now: 0, velocity: 0 }))
  }

  protected prepareInput(ctx: RawTapContext): UCPredictInput {
    return {
      time: ctx.now,
      velocity:
        ctx.velocity == null
          ? Math.floor(Math.random() * 41) + 60
          : ctx.velocity
    };
  }
}

class HandInferenceEngine extends BaseInferenceEngine<HandPredictInput> {
  readonly kind = 'hand';

  constructor() {
    super();
    if (!window.my?.UCTapWrapper) throw new Error("Not loaded");
    this.tapper = new window.my.HandTapWrapper();
  }

  async selfTest() {
    if (window.my?.testHand && !modelTestStatus.hand) {
      await window.my.testHand();
      modelTestStatus.hand = true;
      console.debug("Hand model self-test passed");
    }
  }

  protected initInference() {
    this.predict(this.prepareInput({ now: 0, velocity: 0 }))
  }

  protected prepareInput(ctx: RawTapContext): HandPredictInput {
    // Hand: default right(1), boundary: {6 U H N} -> all produce note >= MIDI_65
    const hand = ctx.pitch === null ? 1 : (ctx.pitch >= 65 ? 1 : 0)
    console.debug("Hand: input hand is", hand)
    return {
      time: ctx.now,
      velocity:
        ctx.velocity == null
          ? Math.floor(Math.random() * 41) + 60
          : ctx.velocity,
      hand: hand
    };
  }
}

class DummyInferenceEngine extends BaseInferenceEngine<DummyInput> {
  readonly kind = 'dummy';
  constructor() {
    super();
  }
  async load() { return }
  async selfTest() { modelTestStatus.dummy = true; }
  reset() { return }
  updateNoteoff(noteoffTime: number) { return }
  dispose() { return }

  protected initInference() { return }

  protected prepareInput(ctx: DummyInput): DummyInput {
    return {
      pitch: ctx.pitch === null
        ? Math.floor(Math.random() * 88) + 21
        : ctx.pitch
    };
  }

  protected predict(input: DummyInput): number {
    return input.pitch;
  }
}

export const engineMap = {
  uc: {
    factory: () => new UCInferenceEngine(),
    label: "UC Mode",
    icon: UserCircle,
  },
  hand: {
    factory: () => new HandInferenceEngine(),
    label: "Hand Condition",
    icon: Hand,
  },
  dummy: {
    factory: () => new DummyInferenceEngine(),
    label: "Dummy",
    icon: Layers,
  },
  experimental: {
    factory: null,
    label: "Experimental",
    icon: Layers,
  },
} as const;

export type InferenceSubMode = keyof typeof engineMap;

export class InferenceFactory {
  static create(mode: InferenceSubMode, initialConfig: InferenceConfig = { samplingType: 'temperature', temperature: 0.8, topP: 0.85 }) {
    const factory = engineMap[mode].factory;
    let engine = factory ? factory() : null;
    if (engine) engine.updateConfig(initialConfig);
    return engine
  }
}

