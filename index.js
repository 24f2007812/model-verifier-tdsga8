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
        !req.body.policy || typeof req.body.policy !== 'object' || Array.isArray(req.body.policy) ||
        !req.body.files || typeof req.body.files !== 'object' || Array.isArray(req.body.files)) {
        return res.status(400).json({ error: "INVALID_INPUT" });
    }

    const { policy, files } = req.body;
    let violations = new Set();
    const addV = (code) => violations.add(code);

    // 1. Policy Validation
    let policyValid = true;
    if (!Array.isArray(policy.requiredSlices) || policy.requiredSlices.length === 0) {
        policyValid = false;
    } else {
        if (policy.requiredSlices.some(s => typeof s !== 'string' || s === '')) policyValid = false;
        if (new Set(policy.requiredSlices).size !== policy.requiredSlices.length) policyValid = false;
    }
    if (typeof policy.license !== 'string' || policy.license === '') policyValid = false;
    if (typeof policy.intendedUse !== 'string' || policy.intendedUse === '') policyValid = false;
    if (typeof policy.limitations !== 'string' || policy.limitations === '') policyValid = false;
    if (!policyValid) addV('INVALID_POLICY');

    // 2. File Presence & Untracked/Unsafe
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

    // 3. Inventory Computation (Hashes ALL present string files except itself)
    let recomputedInventory = [];
    const filesToHash = fileNames.filter(f => f !== 'inventory.json' && typeof files[f] === 'string');
    filesToHash.sort((a, b) => Buffer.from(a).compare(Buffer.from(b)));

    for (const f of filesToHash) {
        // Pushing strictly in name,bytes,sha256 order guarantees correct JSON.stringify order
        recomputedInventory.push({
            name: f,
            bytes: byteLength(files[f]),
            sha256: sha256(files[f])
        });
    }
    const exactCompactJson = JSON.stringify(recomputedInventory);
    const inventoryDigest = sha256(exactCompactJson);

    if ('inventory.json' in files && typeof files['inventory.json'] === 'string') {
        try {
            JSON.parse(files['inventory.json']);
            if (files['inventory.json'] !== exactCompactJson) {
                addV('INVENTORY_MISMATCH');
            }
        } catch (e) {
            addV('INVALID_JSON:inventory.json');
        }
    }

    // 4. Adapter Config
    if ('adapter_config.json' in files && typeof files['adapter_config.json'] === 'string') {
        try {
            const config = JSON.parse(files['adapter_config.json']);
            if (typeof config !== 'object' || config === null || Array.isArray(config)) {
                addV('INVALID_ADAPTER_CONFIG');
            } else {
                const r = config.r;
                const tm = config.target_modules;
                let valid = true;
                if (!Number.isSafeInteger(r) || r <= 0) valid = false;
                if (!Array.isArray(tm) || tm.length === 0 || tm.some(x => typeof x !== 'string' || x === '')) valid = false;
                if (valid && new Set(tm).size !== tm.length) valid = false;
                if (!valid) addV('INVALID_ADAPTER_CONFIG');
            }
        } catch (e) {
            addV('INVALID_JSON:adapter_config.json');
        }
    }

    // 5. Training Manifest
    let parsedManifest = null;
    if ('training_manifest.json' in files && typeof files['training_manifest.json'] === 'string') {
        try {
            parsedManifest = JSON.parse(files['training_manifest.json']);
            if (typeof parsedManifest !== 'object' || parsedManifest === null || Array.isArray(parsedManifest)) {
                addV('INVALID_TRAINING_MANIFEST');
                parsedManifest = null;
            } else {
                if (typeof parsedManifest.base !== 'string' || !/^[a-f0-9]{40}$/.test(parsedManifest.base)) {
                    addV('MUTABLE_BASE_REVISION');
                }
                const manifestReqs = ['task', 'datasetDigest', 'codeDigest', 'trainingConfigDigest', 'modelArtifactDigest', 'evaluationArtifactDigest'];
                for (const mf of manifestReqs) {
                    if (typeof parsedManifest[mf] !== 'string' || parsedManifest[mf] === '') {
                        addV(`MISSING_MANIFEST_FIELD:${mf}`);
                    }
                }
                if (parsedManifest.modelArtifactDigest && typeof files['adapter_model.safetensors'] === 'string') {
                    if (parsedManifest.modelArtifactDigest !== sha256(files['adapter_model.safetensors'])) {
                        addV('MODEL_ARTIFACT_MISMATCH');
                    }
                }
                if (parsedManifest.evaluationArtifactDigest && typeof files['evaluation.json'] === 'string') {
                    if (parsedManifest.evaluationArtifactDigest !== sha256(files['evaluation.json'])) {
                        addV('EVALUATION_DIGEST_MISMATCH');
                    }
                }
            }
        } catch (e) {
            addV('INVALID_JSON:training_manifest.json');
        }
    }

    // 6. Evaluation Validation
    if ('evaluation.json' in files && typeof files['evaluation.json'] === 'string') {
        try {
            const parsedEval = JSON.parse(files['evaluation.json']);
            if (typeof parsedEval !== 'object' || parsedEval === null || Array.isArray(parsedEval)) {
                addV('INVALID_EVALUATION');
            } else {
                let expectedModelDigest = null;
                if (typeof files['adapter_model.safetensors'] === 'string') {
                    expectedModelDigest = sha256(files['adapter_model.safetensors']);
                } else if (parsedManifest && parsedManifest.modelArtifactDigest) {
                    expectedModelDigest = parsedManifest.modelArtifactDigest;
                }
                
                if (!parsedEval.modelArtifactDigest || (expectedModelDigest && parsedEval.modelArtifactDigest !== expectedModelDigest)) {
                    addV('EVALUATION_ARTIFACT_MISMATCH');
                }

                if (typeof parsedEval.aggregate !== 'number' || !Number.isFinite(parsedEval.aggregate) || parsedEval.aggregate < 0 || parsedEval.aggregate > 1) {
                    addV('INVALID_AGGREGATE');
                }

                if (policyValid && Array.isArray(policy.requiredSlices)) {
                    for (const slice of policy.requiredSlices) {
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
            }
        } catch (e) {
            addV('INVALID_JSON:evaluation.json');
        }
    }

    // 7. Model Card Validation
    if ('README.md' in files && typeof files['README.md'] === 'string') {
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
                break; // Missing ending delimiter
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
                    const mManifest = parsedManifest || {}; 
                    
                    if (card.task !== mManifest.task) mismatch = true;
                    if (card.baseRevision !== mManifest.base) mismatch = true;
                    if (card.datasetDigest !== mManifest.datasetDigest) mismatch = true;
                    if (card.modelArtifactDigest !== mManifest.modelArtifactDigest) mismatch = true;
                    
                    // Always validate policy mismatch regardless of valid manifest or not
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

    // 8. Array string output serialization (Strict Buffer Compare)
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