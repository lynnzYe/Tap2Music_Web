export type InferenceSubMode = 'uc' | 'hand' | 'experimental' | 'none';

export abstract class BaseInferenceEngine {
  abstract predict(time: number, velocity: number): number;
  public tapper: any;

  async load() { await this.tapper.init() }
  reset() { this.tapper.reset() }
  updateNoteoff(noteoffTime: number) { this.tapper.updateDur(noteoffTime) }
  // Cleanup resources if needed
  dispose() {
    console.debug(`Disposing engine: ${this.constructor.name}`);
    this.tapper.dispose()
  }
}

export class UCInferenceEngine extends BaseInferenceEngine {
  constructor() {
    super()
    if (!window.my?.UCTapConverter) throw new Error("Not loaded");
    this.tapper = new window.my.UCTapConverter();
  }
  predict(time: number, velocity: number): number {
    return this.tapper.predict(time, velocity)
  }
}

export class InferenceFactory {
  static create(mode: InferenceSubMode): BaseInferenceEngine | null {
    switch (mode) {
      case 'uc': return new UCInferenceEngine();
      // case 'hand': return new HandConditionEngine();
      // case 'experimental': return new ExperimentalEngine();
      default: return null;
    }
  }
}