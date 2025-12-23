function nucleusSample(logits, p = 0.15) {
  // console.debug("Nucleus Sampling")
  return tf.tidy(() => {
    let probs = tf.softmax(logits).squeeze();

    const { values, indices } = tf.topk(probs, probs.shape[0]);
    let cumsum = tf.cumsum(values);

    let mask = cumsum.lessEqual(p);
    mask = mask.logicalOr(tf.oneHot(0, mask.shape[0]).cast("bool"));

    let filtered = values.mul(mask.cast("float32"));
    filtered = filtered.div(filtered.sum());

    const sampled = tf.multinomial(tf.log(filtered), 1).squeeze();
    const ptTensor = indices.gather(sampled);

    const pitchIdx = ptTensor.dataSync()[0];
    ptTensor.dispose();
    return pitchIdx;
  });
}

function choice(probs) {
  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) return i;
  }
  return probs.length - 1; // fallback
}

function temperatureSample(logits, temperature = 1.0) {
  // Convert tensor to array
  const logitsArray = logits.dataSync(); // or logits.arraySync() if multidim

  // 1. Mask out the mask token (109)
  // logitsArray[109] = -1e9;

  // 2. Apply temperature
  const expLogits = logitsArray.map((l) => Math.exp(l / temperature));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  const probs = expLogits.map((e) => e / sumExp);

  // 3. Random choice based on probs
  return choice(probs);
}

class BaseTapEngine {
  constructor() {
    if (new.target === BaseTapEngine) {
      throw new Error("Cannot instantiate abstract class BaseTapEngine");
    }
    if (this.predict === BaseTapEngine.prototype.predict) {
      throw new Error("Subclasses must implement predict()");
    }
    this.dec = null;
    // Performance state
    this.lastTime = null;
    this.lastDur = 0;
    this.lastPitchIdx = 88;
    this.lastHidden = null;
  }
  predict(_) {
    throw new Error("predict() must be implemented");
  }

  async init() {
    await this.dec.init();
    this.reset();
  }

  reset() {
    // console.debug("reset LSTM");
    if (this.lastHidden !== null) {
      this.lastHidden.dispose();
    }
    this.lastTime = null;
    this.lastPitchIdx = 88;
    this.lastHidden = null;
  }

  dispose() {
    if (this.lastHidden !== null) {
      this.lastHidden.dispose();
    }
    this.dec.dispose();
  }

  updateDur(noteoff_time) {
    if (noteoff_time < this.lastTime) {
      console.warn("TapConv cannot have duration less than zero");
    }
    this.lastDur = Math.max(noteoff_time - this.lastTime, 0);
    // console.debug("TapConv. last dur:", this.lastDur);
  }
}

class UCTapEngine extends BaseTapEngine {
  constructor() {
    super();
    // Model
    this.dec = new my.UCModel();
  }

  predict({ time, velocity = null }) {
    // Check inputs
    if (velocity === null) {
    }
    const start = performance.now();
    velocity = velocity === undefined ? 64 : velocity;
    let deltaTime =
      this.lastTime === null ? 0 : (time - this.lastTime) / 1000.0;
    let lastDur = Math.min(this.lastDur, deltaTime);

    if (deltaTime < 0) {
      console.log("Warning: Specified time is in the past");
      deltaTime = 0;
    }
    if (this.lastPitchIdx < 0 || this.lastPitchIdx >= my.PIANO_NUM_KEYS + 1) {
      throw new Error("Specified MIDI note is out of piano's range");
    }

    const log1pDeltaTime = Math.log1p(deltaTime);
    const log1pDur = Math.log1p(lastDur);

    // Run model
    const prevHidden = this.lastHidden;
    if (this.lastPitchIdx === null) {
      this.lastPitchIdx = 88; // start token
    }
    const [pitchIdx, hidden] = tf.tidy(() => {
      // Pitch within 88 classes
      let feat = tf.tensor(
        [[this.lastPitchIdx, log1pDeltaTime, log1pDur, velocity]],
        [1, 4],
        "float32"
      );
      const [plgt, hi] = this.dec.forward(feat, prevHidden);

      const pitchIdx = temperatureSample(plgt);

      return [pitchIdx, hi];
    });

    // Update state
    const end = performance.now();
    const inferTime = ((end - start) / 1000).toFixed(3);
    if (prevHidden !== null) prevHidden.dispose();
    console.debug("Tap2Music:", `🎶 ${pitchIdx + 21}`, `⌚ ${inferTime}s`);
    this.lastPitchIdx = pitchIdx;
    this.lastTime = time;
    this.lastHidden = hidden;
    return pitchIdx + 21;
  }
}

(function (tf, my) {
  my.UCTapEngine = UCTapEngine;
})(window.tf, window.my);
