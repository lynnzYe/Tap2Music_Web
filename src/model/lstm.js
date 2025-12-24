// Resuing code from Chris' https://github.com/chrisdonahue/music-cocreation-tutorial/blob/main/part-2-js-interaction/modules.js

window.my = window.my || {};

(function (tf, my) {
  const PIANO_NUM_KEYS = 88;
  const testThres = 0.03;
  const UC_CKPT_DIR = `ucmodel`;
  const Hand_CKPT_DIR = `handmodel`;
  const SEQ_LEN = 128;

  class Module {
    constructor() {
      this._params = null;
    }

    async init(paramsDir) {
      // Load parameters
      this.dispose();
      const manifest = await fetch(`${paramsDir}/weights_manifest.json`);
      const manifestJson = await manifest.json();
      this._params = await tf.io.loadWeights(manifestJson, `${paramsDir}`);
    }

    dispose() {
      // Dispose of parameters
      if (this._params !== null) {
        for (const n in this._params) {
          this._params[n].dispose();
        }
        this._params = null;
      }
    }
  }

  class LSTMHiddenState {
    constructor(c, h) {
      if (c.length !== h.length) throw "Invalid shapes";
      this.c = c;
      this.h = h;
    }

    dispose() {
      for (let i = 0; i < this.c.length; ++i) {
        this.c[i].dispose();
        this.h[i].dispose();
      }
    }
  }

  function pyTorchLSTMCellFactory(
    kernelInputHidden,
    kernelHiddenHidden,
    biasInputHidden,
    biasHiddenHidden
  ) {
    // Patch between differences in LSTM APIs for PyTorch/Tensorflow
    // NOTE: Fixes kernel packing order
    // PyTorch packs kernel as [i, f, j, o] and Tensorflow [i, j, f, o]
    // References:
    // https://github.com/tensorflow/tfjs/blob/31fd388daab4b21c96b2cb73c098456e88790321/tfjs-core/src/ops/basic_lstm_cell.ts#L47-L78
    // https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
    return (data, c, h) => {
      // NOTE: Modified from Tensorflow.JS basicLSTMCell (see first reference)
      // Create empty forgetBias
      const forgetBias = tf.scalar(0, "float32");

      // Pack kernel
      const kernel = tf.transpose(
        tf.concat([kernelInputHidden, kernelHiddenHidden], 1)
      );

      // Pack bias
      // NOTE: Not sure why PyTorch breaks bias into two terms...
      const bias = tf.add(biasInputHidden, biasHiddenHidden);

      const combined = tf.concat([data, h], 1);
      const weighted = tf.matMul(combined, kernel);
      const res = tf.add(weighted, bias);

      // i = input_gate, j = new_input, f = forget_gate, o = output_gate
      const batchSize = res.shape[0];
      const sliceCols = res.shape[1] / 4;
      const sliceSize = [batchSize, sliceCols];
      const i = tf.slice(res, [0, 0], sliceSize);
      //const j = tf.slice(res, [0, sliceCols], sliceSize);
      //const f = tf.slice(res, [0, sliceCols * 2], sliceSize);
      const f = tf.slice(res, [0, sliceCols], sliceSize);
      const j = tf.slice(res, [0, sliceCols * 2], sliceSize);
      const o = tf.slice(res, [0, sliceCols * 3], sliceSize);

      const newC = tf.add(
        tf.mul(tf.sigmoid(i), tf.tanh(j)),
        tf.mul(c, tf.sigmoid(tf.add(forgetBias, f)))
      );
      const newH = tf.mul(tf.tanh(newC), tf.sigmoid(o));
      return [newC, newH];
    };
  }
  function gelu(x) {
    const c = tf.scalar(Math.sqrt(2 / Math.PI));
    return tf.mul(
      0.5,
      tf.mul(
        x,
        tf.add(1, tf.tanh(tf.mul(c, tf.add(x, tf.mul(0.044715, tf.pow(x, 3))))))
      )
    );
  }

  class UCModel extends Module {
    /*
     * - python has much more hyperparameters. For simplicity did not include those as params.
     * - you'll need to manually modify the code if you want to load a different model.
     */
    constructor() {
      super();
      this.nPitches = PIANO_NUM_KEYS + 1;
      this.pitchEmb = 32;
      this.rnnDim = 128;
      this.rnnNumLayers = 2;
      this._cells = null;
    }

    async init(paramsDir) {
      await super.init(paramsDir === undefined ? UC_CKPT_DIR : paramsDir);

      // Create LSTM cell closures
      this._cells = [];
      for (let l = 0; l < this.rnnNumLayers; ++l) {
        const wih = this._params[`model.lstm.weight_ih_l${l}`];
        if (!wih) {
          throw new Error(`Missing LSTM weights for layer ${l}`);
        }

        this._cells.push(
          pyTorchLSTMCellFactory(
            this._params[`model.lstm.weight_ih_l${l}`],
            this._params[`model.lstm.weight_hh_l${l}`],
            this._params[`model.lstm.bias_ih_l${l}`],
            this._params[`model.lstm.bias_hh_l${l}`]
          )
        );
      }
    }

    initHidden(batchSize) {
      // NOTE: This allocates memory that must later be freed
      const c = [];
      const h = [];
      for (let i = 0; i < this.rnnNumLayers; ++i) {
        c.push(tf.zeros([batchSize, this.rnnDim], "float32"));
        h.push(tf.zeros([batchSize, this.rnnDim], "float32"));
      }
      return new LSTMHiddenState(c, h);
    }

    forward(feat, hx = null) {
      return tf.tidy(() => {
        // feat: (B, 4) -> [pitch, dt, dur, vel]
        if (hx === null) hx = this.initHidden(feat.shape[0]);

        // --- Extract features ---
        const pitchIdx = feat.gather([0], 1).reshape([-1]).toInt();
        const dt = feat.gather([1], 1);
        const dur = feat.gather([2], 1);
        const vel = feat.gather([3], 1);

        // --- Pitch embedding ---
        const pitchEmb = tf.gather(
          this._params["model.pitch_emb.weight"], // (89,32)
          pitchIdx
        ); // (B,32)

        // --- Concatenate inputs (dim = 35) ---
        let x = tf.concat([pitchEmb, dt, dur, vel], 1); // (B,35)

        // --- Input projection ---
        // Linear: xW^T + b
        x = tf.add(
          tf.matMul(x, this._params["model.input_linear.weight"], false, true),
          this._params["model.input_linear.bias"]
        ); // (B,128)

        // --- LSTM ---
        let c = hx.c.slice();
        let h = hx.h.slice();

        for (let l = 0; l < this.rnnNumLayers; ++l) {
          [c[l], h[l]] = this._cells[l](x, c[l], h[l]);
          x = h[l];
        }

        // --- Output head ---
        let y = tf.add(
          tf.matMul(x, this._params["model.out_head.0.weight"], false, true),
          this._params["model.out_head.0.bias"]
        );
        y = tf.relu(y);

        y = tf.add(
          tf.matMul(y, this._params["model.out_head.3.weight"], false, true),
          this._params["model.out_head.3.bias"]
        ); // (B,89)

        return [y, new LSTMHiddenState(c, h)];
      });
    }
  }

  class HandModel {
    constructor({
      handEmbDim = 32,
      filmHiddenDim = 128,
      numHands = 2,
      multiLayerFilm = True,
    } = {}) {
      super();

      this.handEmbDim = handEmbDim;
      this.filmHiddenDim = filmHiddenDim;
      this.numHands = numHands;
      this.multiLayerFilm = multiLayerFilm;
    }

    async init(paramsDir) {
      await super.init(paramsDir === undefined ? UC_CKPT_DIR : paramsDir);

      // Create LSTM cell closures
      this._cells = [];
      for (let l = 0; l < this.rnnNumLayers; ++l) {
        const wih = this._params[`model.uc_core.lstm.weight_ih_l${l}`];
        if (!wih) {
          throw new Error(`Missing LSTM weights for layer ${l}`);
        }

        this._cells.push(
          pyTorchLSTMCellFactory(
            this._params[`model.uc_core.lstm.weight_ih_l${l}`],
            this._params[`model.uc_core.lstm.weight_hh_l${l}`],
            this._params[`model.uc_core.lstm.bias_ih_l${l}`],
            this._params[`model.uc_core.lstm.bias_hh_l${l}`]
          )
        );
      }
    }

    initHidden(batchSize) {
      // NOTE: This allocates memory that must later be freed
      const c = [];
      const h = [];
      for (let i = 0; i < this.rnnNumLayers; ++i) {
        c.push(tf.zeros([batchSize, this.rnnDim], "float32"));
        h.push(tf.zeros([batchSize, this.rnnDim], "float32"));
      }
      return new LSTMHiddenState(c, h);
    }

    forward(feat, hx = null) {
      return tf.tidy(() => {
        // feat: (B,5) [pitch, dt, dur, vel, hand]
        if (hx === null) hx = this.initHidden(feat.shape[0]);

        // --- Split input ---
        const pitchIdx = feat.gather([0], 1).reshape([-1]).toInt();
        const dt = feat.gather([1], 1);
        const dur = feat.gather([2], 1);
        const vel = feat.gather([3], 1);
        const handIdx = feat.gather([4], 1).reshape([-1]).toInt();

        // --- Pitch embedding ---
        const pitchEmb = tf.gather(
          this._params["model.uc_core.pitch_emb.weight"],
          pitchIdx
        );

        // --- Hand embedding ---
        const handEmb = tf.gather(
          this._params["model.hand_emb.weight"],
          handIdx
        ); // (B, handEmbDim)

        // --- Input concat ---
        let x = tf.concat([pitchEmb, dt, dur, vel], 1);

        // --- Input projection ---
        x = tf.add(
          tf.matMul(
            x,
            this._params["model.uc_core.input_linear.weight"],
            false,
            true
          ),
          this._params["model.uc_core.input_linear.bias"]
        ); // (B, hidden)

        // --- Optional input FiLM ---
        if (this.multiLayerFilm) {
          let z = tf.add(
            tf.matMul(
              handEmb,
              this._params["model.input_film_mlp.0.weight"],
              false,
              true
            ),
            this._params["model.input_film_mlp.0.bias"]
          );
          z = gelu(z);

          z = tf.add(
            tf.matMul(
              z,
              this._params["model.input_film_mlp.2.weight"],
              false,
              true
            ),
            this._params["model.input_film_mlp.2.bias"]
          );

          const [gammaIn, betaIn] = tf.split(z, 2, 1);
          x = tf.add(tf.mul(gammaIn, x), betaIn);
        }

        // --- LSTM ---
        let c = hx.c.slice();
        let h = hx.h.slice();

        for (let l = 0; l < this.rnnNumLayers; ++l) {
          [c[l], h[l]] = this._cells[l](x, c[l], h[l]);
          x = h[l];
        }

        // --- FiLM (output) ---
        let f = tf.add(
          tf.matMul(
            handEmb,
            this._params["model.film_mlp.0.weight"],
            false,
            true
          ),
          this._params["model.film_mlp.0.bias"]
        );
        f = gelu(f);

        f = tf.add(
          tf.matMul(f, this._params["model.film_mlp.3.weight"], false, true),
          this._params["model.film_mlp.3.bias"]
        );
        f = gelu(f);

        f = tf.add(
          tf.matMul(f, this._params["model.film_mlp.5.weight"], false, true),
          this._params["model.film_mlp.5.bias"]
        );

        const [gamma, beta] = tf.split(f, 2, 1);
        const xCond = tf.add(tf.mul(gamma, x), beta);

        // --- Output head ---
        let y = tf.add(
          tf.matMul(
            xCond,
            this._params["model.out_head.0.weight"],
            false,
            true
          ),
          this._params["model.out_head.0.bias"]
        );
        y = tf.relu(y);

        y = tf.add(
          tf.matMul(y, this._params["model.out_head.3.weight"], false, true),
          this._params["model.out_head.3.bias"]
        );

        return [y, new LSTMHiddenState(c, h)];
      });
    }
  }

  async function testUCTap() {
    console.log("Start UC Tap test");
    const numBytesBefore = tf.memory().numBytes;

    // Create model
    const decoder = new UCModel();
    await decoder.init();

    // Fetch test case
    const t = await fetch(`${UC_CKPT_DIR}/test.json`).then((r) => r.json());

    // Run test
    let totalErr = 0;
    let him1 = null;
    for (let i = 0; i < 128; ++i) {
      him1 = tf.tidy(() => {
        const row = t.feats[i];
        const feats = tf.tensor(
          [[row[0], row[1], row[2], row[3]]],
          [1, 4],
          "float32"
        );
        const [pitch_logits, hi] = decoder.forward(feats, him1);

        const expectedLogits = tf.tensor(
          [t["pitch_logits"][i]], // wrap in batch dim
          [1, 89],
          "float32"
        );

        const err = tf
          .sum(tf.abs(tf.sub(pitch_logits, expectedLogits)))
          .arraySync();
        totalErr += err;

        if (him1 !== null) him1.dispose();
        return hi;
      });
    }

    // Check equivalence to expected outputs
    if (isNaN(totalErr) || totalErr > testThres) {
      // was 0.015
      console.log("Test failed with error=", totalErr);
      throw new Error("Failed test");
    } else if (totalErr > 0.015) {
      console.log("Warning: total decoder error is", totalErr);
    }

    // Check for memory leaks
    him1.dispose();
    decoder.dispose();
    if (tf.memory().numBytes !== numBytesBefore) {
      console.warn(
        "Memory difference found:",
        tf.memory().numBytes - numBytesBefore
      );
      //   throw "Memory leak";
    }
    console.log("Passed decoder test with total err=", totalErr);
  }

  async function testHand() {
    console.log("Start Hand model test");
    const numBytesBefore = tf.memory().numBytes;

    // Create model
    const decoder = new HandModel();
    await decoder.init();

    // Fetch test case
    const t = await fetch(`${Hand_CKPT_DIR}/test.json`).then((r) => r.json());

    // Run test
    let totalErr = 0;
    let him1 = null;
    for (let i = 0; i < 128; ++i) {
      him1 = tf.tidy(() => {
        const row = t.feats[i];
        const feats = tf.tensor(
          [[row[0], row[1], row[2], row[3], row[4]]],
          [1, 5],
          "float32"
        );
        const [pitch_logits, hi] = decoder.forward(feats, him1);

        const expectedLogits = tf.tensor(
          [t["pitch_logits"][i]], // wrap in batch dim
          [1, 89],
          "float32"
        );

        const err = tf
          .sum(tf.abs(tf.sub(pitch_logits, expectedLogits)))
          .arraySync();
        totalErr += err;

        if (him1 !== null) him1.dispose();
        return hi;
      });
    }

    // Check equivalence to expected outputs
    if (isNaN(totalErr) || totalErr > testThres) {
      // was 0.015
      console.log("Test failed with error=", totalErr);
      throw new Error("Failed test");
    } else if (totalErr > 0.015) {
      console.log("Warning: total decoder error is", totalErr);
    }

    // Check for memory leaks
    him1.dispose();
    decoder.dispose();
    if (tf.memory().numBytes !== numBytesBefore) {
      console.warn(
        "Memory difference found:",
        tf.memory().numBytes - numBytesBefore
      );
      //   throw "Memory leak";
    }
    console.log("Passed decoder test with total err=", totalErr);
  }

  my.PIANO_NUM_KEYS = PIANO_NUM_KEYS;
  my.SEQ_LEN = SEQ_LEN;
  my.UCModel = UCModel;
  my.HandModel = HandModel;
  my.testUCTap = testUCTap;
  my.testHand = testHand;
})(window.tf, window.my);
