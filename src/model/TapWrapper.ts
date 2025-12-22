import * as tf from "@tensorflow/tfjs";

class TapWrapper {
    private tapper: any;

    constructor() {
        if (!window.my?.BeatTracker) throw new Error("Not loaded");
        this.tapper = new window.my.TapConverter();
    }

    async load() { await this.tapper.init(); }
    predict(data: Float32Array) { return this.tapper.predict(data); }

    track(time: number, pitch: number, velocity: number, dbHint: boolean = false): [tf.Tensor, tf.Tensor] {
        return this.tapper.track(time, pitch, velocity, dbHint)
    }

    reset() {
        console.info("reset LSTM beat tracker")
        this.tapper.reset()
    }
}

export default TapWrapper;