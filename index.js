const express = require('express');
const crypto = require('crypto');

const app = express();

// Set high limit for large model string payloads
app.use(express.json({ limit: '50mb' }));

// Strictly catch malformed root JSON payloads
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

app.post('/verify-bundle', (req, res) => {
    // 1. Core Input Shape Validation
    if (!req.body || typeof req.body !== 'object' || Array.isArray(req.body) ||
        !req.body.policy || typeof req.body.policy !== 'object' || Array.isArray(req.body.policy) ||
        !req.body.files || typeof req.body.files !== 'object' || Array.isArray(req.body.files)) {
        return res.status(400).json({ error: "INVALID_INPUT" });
    }

    const { policy, files } = req.body;
    let violations = new Set();
    const addV = (code) => violations.add(code);

    // 2. Policy Validation
    let policyValid = true;
    if (!Array.isArray(policy.requiredSlices) || policy.requiredSlices.length === 0) {
        policyValid = false;
    } else {
        const uniqueSlices = new Set();
        for (const slice of policy.requiredSlices) {
            if (typeof slice !== 'string' || slice === '') { policyValid = false; break; }
            uniqueSlices.add(slice);
        }
        if (uniqueSlices.size !== policy.requiredSlices.length) policyValid = false;
    }
    if (typeof policy.license !== 'string' || policy.license === '') policyValid = false;
    if (typeof policy.intendedUse !== 'string' || policy.intendedUse === '') policyValid = false;
    if (typeof policy.limitations !== 'string' || policy.limitations === '') policyValid = false;
    if (!policyValid) addV('INVALID_POLICY');

    // 3. File Presence & Untracked/Unsafe Weights
    const fileNames = Object.keys(files);
    for (const reqFile of REQUIRED_FILES) {
        if (!(reqFile in files)) {
            addV(`MISSING_FILE:${reqFile}`);
        } else if (typeof files[reqFile] !== 'string') {
            addV(`INVALID_FILE:${reqFile}`);
        }
    }
    for (const f of fileNames) {
        if (!REQUIRED_FILES.includes(f)) {
            addV('UNTRACKED_FILE');
        }
        if (UNSAFE_EXTENSIONS.some(ext => f.endsWith(ext))) {
            addV('UNSAFE_WEIGHTS');
        }
    }

    // 4. Inventory Recomputation (Manual string building for EXACT key order)
    let filesToHash = fileNames.filter(f => f !== 'inventory.json' && typeof files[f] === 'string');
    // Strict UTF-8 Byte Sort
    filesToHash.sort((a, b) => Buffer.from(a, 'utf8').compare(Buffer.from(b, 'utf8')));

    let recomputedArrayStrs = [];
    for (const f of filesToHash) {
        const buf = Buffer.from(files[f], 'utf8');
        const sha = crypto.createHash('sha256').update(buf).digest('hex').toLowerCase();
        // Manually enforce exact key order: name,bytes,sha256
        recomputedArrayStrs.push(`{"name":${JSON.stringify(f)},"bytes":${buf.length},"sha256":"${sha}"}`);
    }
    const exactCompactJson = `[${recomputedArrayStrs.join(',')}]`;
    const inventoryDigest = crypto.createHash('sha256').update(Buffer.from(exactCompactJson, 'utf8')).digest('hex').toLowerCase();

    if (typeof files['inventory.json'] === 'string') {
        try {
            JSON.parse(files['inventory.json']);
            // Compare strictly against the exact computed UTF-8 string
            if (files['inventory.json'] !== exactCompactJson) {
                addV('INVENTORY_MISMATCH');
            }
        } catch (e) {
            addV('INVALID_JSON:inventory.json');
        }
    }

    // 5. Adapter Config Validation
    let validAdapterConfig = false;
    if (typeof files['adapter_config.json'] === 'string') {
        try {
            const config = JSON.parse(files['adapter_config.json']);
            if (typeof config === 'object' && config !== null && !Array.isArray(config)) {
                if (Number.isSafeInteger(config.r) && config.r > 0) {
                    const tm = config.target_modules;
                    if (Array.isArray(tm) && tm.length > 0 && tm.every(x => typeof x === 'string' && x !== '') && new Set(tm).size === tm.length) {
                        validAdapterConfig = true;
                    }
                }
            }
        } catch (e) {
            addV('INVALID_JSON:adapter_config.json');
        }
    }
    if (!validAdapterConfig) addV('INVALID_ADAPTER_CONFIG');

    // 6. Training Manifest Validation
    let parsedManifest = null;
    let validManifestStruct = false;
    if (typeof files['training_manifest.json'] === 'string') {
        try {
            const tm = JSON.parse(files['training_manifest.json']);
            if (typeof tm === 'object' && tm !== null && !Array.isArray(tm)) {
                validManifestStruct = true;
                parsedManifest = tm;
                
                if (typeof tm.base !== 'string' || !/^[a-f0-9]{40}$/.test(tm.base)) {
                    addV('MUTABLE_BASE_REVISION');
                }
                const manifestReqs = ['task', 'datasetDigest', 'codeDigest', 'trainingConfigDigest', 'modelArtifactDigest', 'evaluationArtifactDigest'];
                for (const mf of manifestReqs) {
                    if (typeof tm[mf] !== 'string' || tm[mf] === '') {
                        addV(`MISSING_MANIFEST_FIELD:${mf}`);
                    }
                }
                if (tm.modelArtifactDigest && typeof files['adapter_model.safetensors'] === 'string') {
                    const modelSha = crypto.createHash('sha256').update(Buffer.from(files['adapter_model.safetensors'], 'utf8')).digest('hex').toLowerCase();
                    if (tm.modelArtifactDigest !== modelSha) addV('MODEL_ARTIFACT_MISMATCH');
                }
                if (tm.evaluationArtifactDigest && typeof files['evaluation.json'] === 'string') {
                    const evalSha = crypto.createHash('sha256').update(Buffer.from(files['evaluation.json'], 'utf8')).digest('hex').toLowerCase();
                    if (tm.evaluationArtifactDigest !== evalSha) addV('EVALUATION_DIGEST_MISMATCH');
                }
            }
        } catch (e) {
            addV('INVALID_JSON:training_manifest.json');
        }
    }
    if (!validManifestStruct) addV('INVALID_TRAINING_MANIFEST');

    // 7. Evaluation Validation
    let validEvalStruct = false;
    if (typeof files['evaluation.json'] === 'string') {
        try {
            const ev = JSON.parse(files['evaluation.json']);
            if (typeof ev === 'object' && ev !== null && !Array.isArray(ev)) {
                validEvalStruct = true;
                
                // Bind to manifest digest, or fallback to file digest if manifest invalid/missing
                let expectedModelDigest = null;
                if (parsedManifest && typeof parsedManifest.modelArtifactDigest === 'string') {
                    expectedModelDigest = parsedManifest.modelArtifactDigest;
                } else if (typeof files['adapter_model.safetensors'] === 'string') {
                    expectedModelDigest = crypto.createHash('sha256').update(Buffer.from(files['adapter_model.safetensors'], 'utf8')).digest('hex').toLowerCase();
                }
                
                if (expectedModelDigest && ev.modelArtifactDigest !== expectedModelDigest) {
                    addV('EVALUATION_ARTIFACT_MISMATCH');
                } else if (!expectedModelDigest && !ev.modelArtifactDigest) {
                    addV('EVALUATION_ARTIFACT_MISMATCH');
                }

                if (typeof ev.aggregate !== 'number' || !Number.isFinite(ev.aggregate) || ev.aggregate < 0 || ev.aggregate > 1) {
                    addV('INVALID_AGGREGATE');
                }

                if (policyValid) {
                    for (const slice of policy.requiredSlices) {
                        if (!(slice in ev)) {
                            addV(`MISSING_SLICE:${slice}`);
                        } else {
                            const val = ev[slice];
                            if (typeof val !== 'number' || !Number.isFinite(val) || val < 0 || val > 1) {
                                addV(`SLICE_RANGE:${slice}`);
                            }
                        }
                    }
                }
            }
        } catch (e) {
            addV('INVALID_JSON:evaluation.json');
        }
    }
    if (!validEvalStruct) addV('INVALID_EVALUATION');

    // 8. Model Card (README.md) Validation
    if (typeof files['README.md'] === 'string') {
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
            addV('MISSING_MODEL_CARD');
            addV('MODEL_CARD_COUNT');
        } else if (markerCount > 1) {
            addV('MODEL_CARD_COUNT');
        } else {
            try {
                const card = JSON.parse(payloads[0]);
                if (typeof card !== 'object' || card === null || Array.isArray(card)) {
                    addV('INVALID_MODEL_CARD');
                } else {
                    let mismatch = false;
                    const mm = parsedManifest || {}; 
                    
                    if (card.task !== mm.task) mismatch = true;
                    if (card.baseRevision !== mm.base) mismatch = true;
                    if (card.datasetDigest !== mm.datasetDigest) mismatch = true;
                    if (card.modelArtifactDigest !== mm.modelArtifactDigest) mismatch = true;
                    
                    if (policyValid) {
                        if (card.license !== policy.license) mismatch = true;
                        if (card.intendedUse !== policy.intendedUse) mismatch = true;
                        if (card.limitations !== policy.limitations) mismatch = true;
                    } else {
                        mismatch = true;
                    }

                    if (mismatch) addV('MODEL_CARD_MISMATCH');
                }
            } catch (e) {
                addV('INVALID_MODEL_CARD');
            }
        }
    }

    // 9. Output Formatting (Strict UTF-8 Array Sort)
    const violationsArray = Array.from(violations).sort((a, b) => Buffer.from(a, 'utf8').compare(Buffer.from(b, 'utf8')));
    
    res.json({
        decision: violationsArray.length === 0 ? "admit" : "reject",
        violations: violationsArray,
        inventoryDigest: inventoryDigest
    });
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Verification service running on port ${PORT}`);
});