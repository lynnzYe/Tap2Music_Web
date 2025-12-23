// global.d.ts or src/types/global.d.ts
export { };

declare global {
    interface Window {
        my?: {
            UCTap2Music?: any
            UCTapConverter?: any
            testUCTap?: (...args: any[]) => Promise<any>;
            [key: string]: any;
        };
        _midiTestRan?: boolean;
    }
}
