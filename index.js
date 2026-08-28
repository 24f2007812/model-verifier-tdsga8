const express = require('express');
const crypto = require('crypto');
const bodyParser = require('body-parser');

const app = express();

app.use(bodyParser.json({ limit: '50mb' }));

app.use((err, req, res, next) => {
    if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
        return res.status(400).json({ error: 'INVALID_INPUT' });
    }
    next();
});

const REQUIRED_FILES = [
    'README.md', 'adapter_config.json', 'adapter_model.safetensors',
    'evaluation.json', 'inventory.json', 'training_manifest.json'
];

const UNSAFE_EXTENSIONS = ['.bin', '.pt', '.pth', '.pkl', '.pickle'];

function sha256(str) {
    return crypto.createHash('sha256').update(str, 'utf8').digest('hex').toLowerCase();
}

function byteLength(str) {
    return Buffer.byteLength(str, 'utf8');
}

app.post('/verify-bundle', (req, res) => {
    if (!req.body || typeof req.body !== 'object' || Array.isArray(req.body) ||
        !req.body.policy || typeof req.body.policy !== 'object' ||
        !req.body.files || typeof req.body.files !== 'object' || Array.isArray(req.body.files)) {
        return res.status(400).json({ error: "INVALID_INPUT" });
    }

    const { policy, files } = req.body;
    let violations = new Set();
    const addV = (code) => violations.add(code);

    // 1. Policy Validation
    let validPolicy = true;
    if (!policy.requiredSlices || !Array.isArray(policy.requiredSlices) || policy.requiredSlices.length === 0) validPolicy = false;
    else if (new Set(policy.requiredSlices).size !== policy.requiredSlices.length) validPolicy = false;
    else if (policy.requiredSlices.some(s => typeof s !== 'string' || s === '')) validPolicy = false;

    if (typeof policy.license !== 'string' || policy.license === '') validPolicy = false;
    if (typeof policy.intendedUse !== 'string' || policy.intendedUse === '') validPolicy = false;
    if (typeof policy.limitations !== 'string' || policy.limitations === '') validPolicy = false;
    
    if (!validPolicy) addV('INVALID_POLICY');

    // 2. File Presence & Untracked/Unsafe Files
    const fileNames = Object.keys(files);
    for (const reqFile of REQUIRED_FILES) {
        if (!fileNames.includes(reqFile)) {
            addV(`MISSING_FILE:${reqFile}`);
        } else if (typeof files[reqFile] !== 'string') {
            addV(`INVALID_FILE:${reqFile}`);
        }
    }

    for (const file of fileNames) {
        if (!REQUIRED_FILES.includes(file)) {
            addV('UNTRACKED_FILE');
        }
        if (UNSAFE_EXTENSIONS.some(ext => file.endsWith(ext))) {
            addV('UNSAFE_WEIGHTS');
        }
    }

    // 3. Inventory Validation & Exact Bytes Recomputation
    let recomputedInventory = [];
    const filesToHash = fileNames.filter(f => f !== 'inventory.json' && typeof files[f] === 'string');
    
    // Exact UTF-8 string sort
    filesToHash.sort((a, b) => Buffer.from(a).compare(Buffer.from(b)));

    for (const f of filesToHash) {
        recomputedInventory.push({
            name: f,
            bytes: byteLength(files[f]),
            sha256: sha256(files[f])
        });
    }

    const exactCompactJson = JSON.stringify(recomputedInventory);
    const inventoryDigest = sha256(exactCompactJson);

    if (files['inventory.json'] && typeof files['inventory.json'] === 'string') {
        try {
            JSON.parse(files['inventory.json']);
            // Must strictly match the recomputed compacted exact JSON payload
            if (files['inventory.json'] !== exactCompactJson) {
                addV('INVENTORY_MISMATCH');
            }
        } catch (e) {
            addV('INVALID_JSON:inventory.json');
        }
    }

    // 4. Adapter Config Validation
    if (files['adapter_config.json'] && typeof files['adapter_config.json'] === 'string') {
        try {
            const config = JSON.parse(files['adapter_config.json']);
            if (typeof config !== 'object' || config === null || Array.isArray(config)) throw new Error();
            
            const r = config.r;
            const tm = config.target_modules;
            
            if (!Number.isSafeInteger(r) || r <= 0) addV('INVALID_ADAPTER_CONFIG');
            else if (!Array.isArray(tm) || tm.length === 0 || tm.some(x => typeof x !== 'string' || x === '') || new Set(tm).size !== tm.length) addV('INVALID_ADAPTER_CONFIG');
        } catch (e) {
            addV('INVALID_JSON:adapter_config.json');
        }
    }

    // 5. Training Manifest Validation & Binding
    let parsedManifest = null;
    if (files['training_manifest.json'] && typeof files['training_manifest.json'] === 'string') {
        try {
            parsedManifest = JSON.parse(files['training_manifest.json']);
            if (typeof parsedManifest !== 'object' || parsedManifest === null || Array.isArray(parsedManifest)) throw new Error();

            if (typeof parsedManifest.base !== 'string' || !/^[a-f0-9]{40}$/.test(parsedManifest.base)) {
                addV('MUTABLE_BASE_REVISION');
            }

            const manifestReqs = ['task', 'datasetDigest', 'codeDigest', 'trainingConfigDigest', 'modelArtifactDigest', 'evaluationArtifactDigest'];
            for (const mf of manifestReqs) {
                if (typeof parsedManifest[mf] !== 'string' || parsedManifest[mf] === '') {
                    addV(`MISSING_MANIFEST_FIELD:${mf}`);
                }
            }

            // Digest comparisons from exact recomputed files
            if (parsedManifest.modelArtifactDigest && typeof files['adapter_model.safetensors'] === 'string') {
                if (parsedManifest.modelArtifactDigest !== sha256(files['adapter_model.safetensors'])) addV('MODEL_ARTIFACT_MISMATCH');
            }
            if (parsedManifest.evaluationArtifactDigest && typeof files['evaluation.json'] === 'string') {
                if (parsedManifest.evaluationArtifactDigest !== sha256(files['evaluation.json'])) addV('EVALUATION_DIGEST_MISMATCH');
            }
        } catch (e) {
            addV('INVALID_JSON:training_manifest.json');
        }
    } else if (!violations.has('MISSING_FILE:training_manifest.json') && !violations.has('INVALID_FILE:training_manifest.json')) {
        addV('INVALID_TRAINING_MANIFEST');
    }

    // 6. Evaluation Binding and Ranges
    if (files['evaluation.json'] && typeof files['evaluation.json'] === 'string') {
        try {
            const parsedEval = JSON.parse(files['evaluation.json']);
            if (typeof parsedEval !== 'object' || parsedEval === null || Array.isArray(parsedEval)) throw new Error();

            const actualModelDigest = typeof files['adapter_model.safetensors'] === 'string' ? sha256(files['adapter_model.safetensors']) : undefined;
            if (actualModelDigest && parsedEval.modelArtifactDigest !== actualModelDigest) {
                addV('EVALUATION_ARTIFACT_MISMATCH');
            }

            if (typeof parsedEval.aggregate !== 'number' || !Number.isFinite(parsedEval.aggregate) || parsedEval.aggregate < 0 || parsedEval.aggregate > 1) {
                addV('INVALID_AGGREGATE');
            }

            if (Array.isArray(policy.requiredSlices)) {
                for (const slice of policy.requiredSlices) {
                    if (typeof slice !== 'string') continue;
                    if (!(slice in parsedEval)) {
                        addV(`MISSING_SLICE:${slice}`);
                    } else {
                        const val = parsedEval[slice];
                        if (typeof val !== 'number' || !Number.isFinite(val) || val < 0 || val > 1) {
                            addV(`SLICE_RANGE:${slice}`);
                        }
                    }
                }
            }
        } catch (e) {
            addV('INVALID_JSON:evaluation.json');
        }
    } else if (!violations.has('MISSING_FILE:evaluation.json') && !violations.has('INVALID_FILE:evaluation.json')) {
        addV('INVALID_EVALUATION');
    }

    // 7. Model Card Validation
    if (files['README.md'] && typeof files['README.md'] === 'string') {
        const readme = files['README.md'];
        const prefix = '<!-- tds-model-card ';
        const suffix = '-->';
        
        let markerCount = 0;
        let lastIndex = 0;
        let payloads = [];

        while ((lastIndex = readme.indexOf(prefix, lastIndex)) !== -1) {
            let endIndex = readme.indexOf(suffix, lastIndex + prefix.length);
            if (endIndex !== -1) {
                markerCount++;
                payloads.push(readme.substring(lastIndex + prefix.length, endIndex));
                lastIndex = endIndex + suffix.length;
            } else {
                break;
            }
        }

        if (markerCount === 0) {
            addV('MODEL_CARD_COUNT');
            addV('MISSING_MODEL_CARD');
        } else if (markerCount > 1) {
            addV('MODEL_CARD_COUNT');
        } else {
            try {
                const card = JSON.parse(payloads[0]);
                if (typeof card !== 'object' || card === null || Array.isArray(card)) {
                    addV('INVALID_MODEL_CARD');
                } else {
                    let cardMismatch = false;
                    
                    if (parsedManifest) {
                        if (card.task !== parsedManifest.task ||
                            card.baseRevision !== parsedManifest.base ||
                            card.datasetDigest !== parsedManifest.datasetDigest ||
                            card.modelArtifactDigest !== parsedManifest.modelArtifactDigest) {
                            cardMismatch = true;
                        }
                    }
                    
                    if (card.license !== policy.license ||
                        card.intendedUse !== policy.intendedUse ||
                        card.limitations !== policy.limitations) {
                        cardMismatch = true;
                    }

                    if (cardMismatch) addV('MODEL_CARD_MISMATCH');
                }
            } catch (e) {
                addV('INVALID_MODEL_CARD');
            }
        }
    }

    // 8. Output Serialization - Ensure UTF-8 byte ordering for keys!
    const violationsArray = Array.from(violations).sort((a, b) => Buffer.from(a).compare(Buffer.from(b)));
    
    res.json({
        decision: violationsArray.length === 0 ? "admit" : "reject",
        violations: violationsArray,
        inventoryDigest: inventoryDigest
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Verification service running on port ${PORT}`);
});