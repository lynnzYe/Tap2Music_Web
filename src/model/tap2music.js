export const BEAT_THRES = 0.4;
export const DOWNBEAT_THRES = 0.5;

(function (tf, my) {
  class TapConverter {
    constructor() {
      // Model
      this.dec = new my.UCTap2Music();

      // Performance state
      this.lastTime = null;
      this.lastDur = 0;
      this.lastPitchIdx = null;
      this.lastHidden = null;
    }

    async init() {
      await this.dec.init();

      // Warm start
      this.track(0, 64);
      this.reset();
    }

    reset() {
      if (this.lastHidden !== null) {
        this.lastHidden.dispose();
      }
      this.lastTime = null;
      this.lastPitchIdx = null;
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
      console.debug("TapConv. last dur:", this.lastDur);
    }

    nucleusSample(logits, p = 0.07) {
      console.debug("Nucleus Sampling")
      return tf.tidy(() => {
        let probs = tf.softmax(logits).squeeze();

        const { values, indices } = tf.topk(probs, probs.shape[0]);
        let cumsum = tf.cumsum(values);

        let mask = cumsum.lessEqual(p);
        mask = mask.logicalOr(tf.oneHot(0, mask.shape[0]).cast("bool"));

        let filtered = values.mul(mask.cast("float32"));
        filtered = filtered.div(filtered.sum());

        const sampled = tf.multinomial(tf.log(filtered), 1).squeeze();
        return indices.gather(sampled);
      });
    }

    track(time, velocity) {
      // Check inputs
      velocity = velocity === undefined ? 64 : velocity;
      let deltaTime = this.lastTime === null ? 1e6 : time - this.lastTime;
      let lastDur = Math.min(this.lastDur, deltaTime);

      if (deltaTime < 0) {
        console.log("Warning: Specified time is in the past");
        deltaTime = 0;
      }
      if (this.lastPitchIdx < 0 || this.lastPitchIdx >= my.PIANO_NUM_KEYS) {
        throw new Error("Specified MIDI note is out of piano's range");
      }

      const log1pDeltaTime = Math.log1p(deltaTime);
      const log1pDur = Math.log1p(lastDur);

      // Run model
      const prevHidden = this.lastHidden;
      const [pitchIdxTensor, hidden] = tf.tidy(() => {
        // Pitch within 88 classes
        let feat = tf.tensor(
          [[this.lastPitchIdx, log1pDeltaTime, log1pDur, velocity]],
          [1, 4],
          "float32"
        );
        const [plgt, hi] = this.dec.forward(feat, prevHidden);

        const pitchIdxTensor = this.nucleusSample(plgt);

        return [pitchIdxTensor, hi];
      });

      const pitchIdx = pitchIdxTensor.dataSync()[0];
      pitchIdxTensor.dispose();

      // Update state
      if (prevHidden !== null) prevHidden.dispose();
      this.lastPitchIdx = pitchIdx;
      this.lastTime = time;
      this.lastHidden = hidden;
      return pitchIdx + 21;
    }
  }
  my.TapConverter = TapConverter;
})(window.tf, window.my);
