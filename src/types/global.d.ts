// global.d.ts or src/types/global.d.ts
export { };

declare global {
    interface Window {
        my?: {
            UCModel?: any
            UCTapEngine?: any
            testUCTap?: (...args: any[]) => Promise<any>;
            [key: string]: any;
        };
        _midiTestRan?: boolean;
    }
}
