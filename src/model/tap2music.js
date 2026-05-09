function nucleusSample(logits, p = 0.7) {
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

class BaseTapWrapper {
  constructor() {
    if (new.target === BaseTapWrapper) {
      throw new Error("Cannot instantiate abstract class BaseTapEngine");
    }
    if (this.predict === BaseTapWrapper.prototype.predict) {
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

class UCTapWrapper extends BaseTapWrapper {
  constructor() {
    super();
    // Model
    this.dec = new my.UCModel();
  }

  predict({
    pitch, // Placeholder for GtInject, usually not used
    time,
    velocity = 64,
    isGtInject = false,
    samplingType = "temperature",
    temperature = 1.1,
    topP = 0.7,
  }) {
    // Check inputs
    const start = performance.now();
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
        "float32",
      );
      const [plgt, hi] = this.dec.forward(feat, prevHidden);

      let pitchIdx = 88;
      if (samplingType == "temperature") {
        pitchIdx = temperatureSample(plgt, temperature);
      } else if (samplingType == "nucleus") {
        pitchIdx = nucleusSample(plgt, topP);
      } else {
        throw new Error("Unknown sampling type:", samplingType);
      }
      return [pitchIdx, hi];
    });

    // Update state
    const end = performance.now();
    const inferTime = ((end - start) / 1000).toFixed(3);
    if (prevHidden !== null) prevHidden.dispose();
    console.debug("Tap2Music:", `🎶 ${pitchIdx + 21}`, `⌚ ${inferTime}s`);

    const finalPitchIdx = isGtInject ? pitch - 21 : pitchIdx;
    console.debug("is inject:", isGtInject);
    this.lastPitchIdx = finalPitchIdx;
    this.lastTime = time;
    this.lastHidden = hidden;
    return pitchIdx + 21;
  }
}

class HandTapWrapper extends BaseTapWrapper {
  constructor() {
    super();
    // Model
    this.dec = new my.HandModel();
  }

  predict({
    time,
    velocity = 64,
    hand = 1,
    isGtInject = false,
    samplingType = "temperature",
    temperature = 1.1,
    topP = 0.7,
  }) {
    // Check inputs
    const start = performance.now();
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
        [[this.lastPitchIdx, log1pDeltaTime, log1pDur, velocity, hand]],
        [1, 5],
        "float32",
      );
      const [plgt, hi] = this.dec.forward(feat, prevHidden);

      let pitchIdx = 88;
      if (samplingType == "temperature") {
        pitchIdx = temperatureSample(plgt, temperature);
      } else if (samplingType == "nucleus") {
        pitchIdx = nucleusSample(plgt, topP);
      } else {
        throw new Error("Unknown sampling type:", samplingType);
      }

      return [pitchIdx, hi];
    });

    // Update state
    const end = performance.now();
    const inferTime = ((end - start) / 1000).toFixed(3);
    if (prevHidden !== null) prevHidden.dispose();
    console.debug("Tap2Music:", `🎶 ${pitchIdx + 21}`, `⌚ ${inferTime}s`);

    const finalPitchIdx = isGtInject ? pitch - 21 : pitchIdx;
    this.lastPitchIdx = finalPitchIdx;
    this.lastTime = time;
    this.lastHidden = hidden;
    return pitchIdx + 21;
  }
}

class RTPTapWrapper extends BaseTapWrapper {
  constructor() {
    super();
    this.dec = new my.RTPModel();
    // History stores { pitch, time } of the user's actual taps
    this.history = [];
  }

  reset() {
    super.reset();
    this.history = [];
  }

  // Calculate rank of currentPitch against the previous nNote taps
  getNRank(currentPitch, nNote = 10) {
    if (this.history.length === 0) return 0;

    // Look at up to nNote - 1 past taps + current pitch = nNote window
    const windowSize = nNote - 1;
    const window = this.history.slice(-windowSize);

    let rank = 0;
    // Count preceding notes >= current pitch (matches lexsort descending)
    for (let i = 0; i < window.length; i++) {
      if (window[i].pitch >= currentPitch) rank++;
    }
    return rank;
  }

  // Calculate time rank of currentPitch against continuous preceding cluster
  getTimeRank(currentPitch, currentTime, timeThresh = 0.05, nNote = 10) {
    if (this.history.length === 0) return 10; // sentinel

    let validNotes = [];
    let prevTime = currentTime;
    const windowSize = nNote - 1;

    // Walk backwards through history to find the continuous time cluster
    // Stop if the gap (ioi) exceeds threshold or we hit the window limit
    for (
      let i = this.history.length - 1;
      i >= Math.max(0, this.history.length - windowSize);
      i--
    ) {
      const histNote = this.history[i];
      const ioi = (prevTime - histNote.time) / 1000.0;

      if (ioi < timeThresh) {
        validNotes.unshift(histNote);
        prevTime = histNote.time;
      } else {
        break;
      }
    }

    if (validNotes.length === 0) return 10; // sentinel

    let rank = 0;
    // Count preceding notes <= current pitch (matches lexsort ascending)
    for (let i = 0; i < validNotes.length; i++) {
      if (validNotes[i].pitch <= currentPitch) rank++;
    }
    return rank;
  }

  predict({
    time,
    velocity = 64,
    pitch, // MUST pass the actual tap pitch now (it doesn't have to be gt, just provide RTP information)
    isGtInject = false,
    samplingType = "temperature", // Or can be refactored to take n-rank and time-rank features
    temperature = 1.1,
    topP = 0.7,
  }) {
    const start = performance.now();
    let deltaTime =
      this.lastTime === null ? 0 : (time - this.lastTime) / 1000.0;
    let lastDur = Math.min(this.lastDur, deltaTime);

    if (deltaTime < 0) {
      console.log("Warning: Specified time is in the past");
      deltaTime = 0;
    }

    // Seed variables if this is the first step
    if (this.lastTime === null) {
      this.lastPitchIdx = 88; // Start token for the network input
    }

    if (this.lastPitchIdx < 0 || this.lastPitchIdx >= my.PIANO_NUM_KEYS + 1) {
      throw new Error("Specified MIDI note is out of piano's range");
    }

    const log1pDeltaTime = Math.log1p(deltaTime);
    const log1pDur = Math.log1p(lastDur);

    // Compute relative positions based on the CURRENT tap pitch vs PAST taps
    const nRank = this.getNRank(pitch, 10);
    const timeRank = this.getTimeRank(pitch, time, 0.05, 10);

    const prevHidden = this.lastHidden;
    const [pitchIdx, hidden] = tf.tidy(() => {
      // Input features exactly match your python feat extraction:
      // [prev_predicted_pitch, log_ioi, log_dur, velocity, n_rank, time_rank]
      let feat = tf.tensor(
        [
          [
            this.lastPitchIdx,
            log1pDeltaTime,
            log1pDur,
            velocity,
            nRank,
            timeRank,
          ],
        ],
        [1, 6],
        "float32",
      );
      const [plgt, hi] = this.dec.forward(feat, prevHidden);

      let pIdx = 88;
      if (samplingType === "temperature") {
        pIdx = temperatureSample(plgt, temperature);
      } else if (samplingType === "nucleus") {
        pIdx = nucleusSample(plgt, topP);
      } else {
        throw new Error("Unknown sampling type:", samplingType);
      }
      return [pIdx, hi];
    });

    const end = performance.now();
    const inferTime = ((end - start) / 1000).toFixed(3);
    if (prevHidden !== null) prevHidden.dispose();
    console.debug(
      "Tap2Music (RTP):",
      `🎶 ${pitchIdx + 21}`,
      `⌚ ${inferTime}s`,
      `| nR: ${nRank}, tR: ${timeRank}`,
    );

    const finalPitchIdx = isGtInject ? pitch - 21 : pitchIdx;

    // 1. Store the PREDICTED pitch for the next step's network input
    this.lastPitchIdx = finalPitchIdx;
    this.lastTime = time;
    this.lastHidden = hidden;

    // 2. Store the TAP pitch in history for the next step's rank calculations
    this.history.push({
      pitch: isGtInject ? pitch : finalPitchIdx + 21,
      time: time,
    });
    // console.debug("this history:", this.history);
    if (this.history.length > 10) {
      this.history.shift();
    }

    return pitchIdx + 21;
  }
}

(function (tf, my) {
  my.UCTapWrapper = UCTapWrapper;
  my.HandTapWrapper = HandTapWrapper;
  my.RTPTapWrapper = RTPTapWrapper;
})(window.tf, window.my);
