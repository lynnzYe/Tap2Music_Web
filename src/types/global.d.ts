// global.d.ts or src/types/global.d.ts
export { };

declare global {
    interface Window {
        my?: {
            // Base LSTM models and tests
            UCModel?: any
            HandModel?: any
            testUCTap?: (...args: any[]) => Promise<any>;
            testHand?: (...args: any[]) => Promise<any>;

            // stateful LSTM wrappers 
            UCTapWrapper?: any
            HandTapWrapper?: any

            // Any other
            [key: string]: any;
        };
        _midiTestRan?: boolean;
    }
}
