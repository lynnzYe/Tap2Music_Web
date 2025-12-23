type UCPredictInput = {
  time: number
  velocity?: number | null
};

type HandRangePredictInput = {
  kind: "text";
  text: string;
  maxLen?: number;
};

type RawTapContext = {
  now: number;
  pitch?: number | null; // currently performed MIDI pitch, used for range/hand condition
  velocity?: number | null;
  chord?: number | null; // for chord-conditioned model, requires further logic to extract chords on the fly
};

type InferenceInputMap = {
  uc: UCPredictInput;
  hand: HandRangePredictInput;
  // experimental: ExperimentalPredictInput;
};

const modelTestStatus: Record<InferenceSubMode, boolean> = {
  uc: false,
  hand: false,
  experimental: false,
  dummy: true
};

export abstract class BaseInferenceEngine<TInput> {
  public tapper: any;
  abstract readonly kind: keyof InferenceInputMap;

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

  // Internel methods to be implemented
  protected abstract initInference(): void; // PAD to mark Start of New Sequence
  protected abstract prepareInput(ctx: RawTapContext): TInput;
  protected abstract predict(input: TInput): number;
}

class UCInferenceEngine extends BaseInferenceEngine<UCPredictInput> {
  readonly kind = 'uc';

  constructor() {
    super();
    if (!window.my?.UCTapEngine) throw new Error("Not loaded");
    this.tapper = new window.my.UCTapEngine();
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

  protected predict(input: UCPredictInput): number {
    return this.tapper.predict(input);
  }
}

const engineMap = {
  uc: () => new UCInferenceEngine(),
  // hand: () => new HandConditionEngine(),
  hand: null,
  experimental: null,
  dummy: null
} as const;

export type InferenceSubMode = keyof typeof engineMap;

export class InferenceFactory {
  static create(mode: InferenceSubMode) {
    const factory = engineMap[mode];
    return factory ? factory() : null;
  }
}

