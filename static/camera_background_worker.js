import {
    FilesetResolver,
    ImageSegmenter,
} from '/static/vendor/mediapipe/tasks-vision/vision_bundle.mjs';

const WASM_ROOT = '/static/vendor/mediapipe/tasks-vision/wasm';
const MODEL_PATH = '/static/vendor/mediapipe/models/selfie_segmenter_landscape.tflite';

let imageSegmenter = null;
let personMaskIndex = 1;

async function initialize() {
    if (imageSegmenter) return;
    if (typeof OffscreenCanvas === 'undefined') {
        throw new Error('OffscreenCanvas is unavailable');
    }

    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT, true);
    imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: MODEL_PATH,
            delegate: 'CPU',
        },
        canvas: new OffscreenCanvas(1, 1),
        runningMode: 'VIDEO',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
    });

    const labels = imageSegmenter.getLabels().map((label) => label.toLowerCase());
    const labelledPerson = labels.findIndex((label) => label.includes('person'));
    if (labelledPerson >= 0) personMaskIndex = labelledPerson;
}

function closeBitmap(bitmap) {
    try {
        bitmap?.close?.();
    } catch (_) {
        // Some browsers release transferred bitmaps automatically.
    }
}

self.addEventListener('message', async (event) => {
    const { type } = event.data || {};

    if (type === 'init') {
        try {
            await initialize();
            self.postMessage({ type: 'ready' });
        } catch (error) {
            self.postMessage({ type: 'error', stage: 'init', message: error?.message || String(error) });
        }
        return;
    }

    if (type === 'segment') {
        const { bitmap, timestamp } = event.data;
        try {
            await initialize();
            imageSegmenter.segmentForVideo(bitmap, timestamp, (result) => {
                const masks = result.confidenceMasks || [];
                const personMask = masks[personMaskIndex] || masks.at(-1);
                if (!personMask) throw new Error('No person mask returned');

                const values = new Float32Array(personMask.getAsFloat32Array());
                self.postMessage({
                    type: 'mask',
                    width: personMask.width,
                    height: personMask.height,
                    values: values.buffer,
                }, [values.buffer]);
            });
        } catch (error) {
            self.postMessage({ type: 'error', stage: 'segment', message: error?.message || String(error) });
        } finally {
            closeBitmap(bitmap);
        }
        return;
    }

    if (type === 'dispose') {
        imageSegmenter?.close?.();
        imageSegmenter = null;
        self.postMessage({ type: 'disposed' });
    }
});
