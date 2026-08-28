const express = require('express');
const crypto = require('crypto');
const bodyParser = require('body-parser');

const app = express();

// Increase payload limit for model strings
app.use(bodyParser.json({ limit: '50mb' }));
// Handle malformed root JSON
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
    // 1. Initial Input Validation
    if (!req.body || typeof req.body !== 'object' || Array.isArray(req.body) ||
        !req.body.policy || typeof req.body.policy !== 'object' ||
        !req.body.files || typeof req.body.files !== 'object' || Array.isArray(req.body.files)) {
        return res.status(400).json({ error: "INVALID_INPUT" });
    }

    const { policy, files } = req.body;
    let violations = new Set();
    let inventoryDigest = "";

    // Helper to add violation
    const addV = (code) => violations.add(code);

    // 2. Policy Validation
    if (!policy.requiredSlices || !Array.isArray(policy.requiredSlices) || policy.requiredSlices.length === 0 ||
        !policy.license || typeof policy.license !== 'string' ||
        !policy.intendedUse || typeof policy.intendedUse !== 'string' ||
        !policy.limitations || typeof policy.limitations !== 'string') {
        addV('INVALID_POLICY');
    }

    // 3. File Presence & Untracked/Unsafe Files
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
            if (UNSAFE_EXTENSIONS.some(ext => file.endsWith(ext))) {
                addV('UNSAFE_WEIGHTS');
            }
        }
    }

    // 4. Inventory Validation & Recomputation
    let recomputedInventory = [];
    const filesToHash = REQUIRED_FILES.filter(f => f !== 'inventory.json');
    filesToHash.sort();

    for (const f of filesToHash) {
        if (files[f] && typeof files[f] === 'string') {
            recomputedInventory.push({
                name: f,
                bytes: byteLength(files[f]),
                sha256: sha256(files[f])
            });
        }
    }

    const exactCompactJson = JSON.stringify(recomputedInventory);
    inventoryDigest = sha256(exactCompactJson);

    let parsedInventory = null;
    if (files['inventory.json'] && typeof files['inventory.json'] === 'string') {
        try {
            parsedInventory = JSON.parse(files['inventory.json']);
            if (!Array.isArray(parsedInventory)) throw new Error();
            
            // Compare items
            const incomingStr = JSON.stringify(parsedInventory);
            if (incomingStr !== exactCompactJson) {
                addV('INVENTORY_MISMATCH');
            }
        } catch (e) {
            addV('INVALID_JSON:inventory.json');
        }
    }

    // 5. Adapter Config Validation
    if (files['adapter_config.json'] && typeof files['adapter_config.json'] === 'string') {
        try {
            const config = JSON.parse(files['adapter_config.json']);
            if (typeof config !== 'object' || config === null || Array.isArray(config)) throw new Error();
            
            const r = config.r;
            const tm = config.target_modules;
            
            let valid = true;
            if (!Number.isSafeInteger(r) || r <= 0) valid = false;
            if (!Array.isArray(tm) || tm.length === 0 || !tm.every(x => typeof x === 'string')) valid = false;
            if (valid && new Set(tm).size !== tm.length) valid = false; // Check unique

            if (!valid) addV('INVALID_ADAPTER_CONFIG');
        } catch (e) {
            addV('INVALID_JSON:adapter_config.json');
        }
    }

    // 6. Training Manifest Validation
    let parsedManifest = null;
    if (files['training_manifest.json'] && typeof files['training_manifest.json'] === 'string') {
        try {
            parsedManifest = JSON.parse(files['training_manifest.json']);
            if (typeof parsedManifest !== 'object' || parsedManifest === null || Array.isArray(parsedManifest)) throw new Error();

            if (!parsedManifest.base || typeof parsedManifest.base !== 'string' || !/^[a-f0-9]{40}$/.test(parsedManifest.base)) {
                addV('MUTABLE_BASE_REVISION');
            }

            const manifestReqs = ['task', 'datasetDigest', 'codeDigest', 'trainingConfigDigest', 'modelArtifactDigest', 'evaluationArtifactDigest'];
            for (const mf of manifestReqs) {
                if (!parsedManifest[mf]) addV(`MISSING_MANIFEST_FIELD:${mf}`);
            }

            // Digest match checks
            if (parsedManifest.modelArtifactDigest && files['adapter_model.safetensors']) {
                if (parsedManifest.modelArtifactDigest !== sha256(files['adapter_model.safetensors'])) {
                    addV('MODEL_ARTIFACT_MISMATCH');
                }
            }
            if (parsedManifest.evaluationArtifactDigest && files['evaluation.json']) {
                if (parsedManifest.evaluationArtifactDigest !== sha256(files['evaluation.json'])) {
                    addV('EVALUATION_DIGEST_MISMATCH');
                }
            }
        } catch (e) {
            addV('INVALID_JSON:training_manifest.json');
        }
    } else {
        addV('INVALID_TRAINING_MANIFEST');
    }

    // 7. Evaluation Validation
    let parsedEval = null;
    if (files['evaluation.json'] && typeof files['evaluation.json'] === 'string') {
        try {
            parsedEval = JSON.parse(files['evaluation.json']);
            if (typeof parsedEval !== 'object' || parsedEval === null || Array.isArray(parsedEval)) throw new Error();

            // Binding check
            if (parsedManifest && parsedManifest.modelArtifactDigest) {
                if (parsedEval.modelArtifactDigest !== parsedManifest.modelArtifactDigest) {
                    addV('EVALUATION_ARTIFACT_MISMATCH');
                }
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
    } else {
         addV('INVALID_EVALUATION');
    }

    // 8. Model Card (README.md) Validation
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
                break; // Malformed boundary
            }
        }

        if (markerCount === 0) {
            addV('MODEL_CARD_COUNT');
            addV('MISSING_MODEL_CARD');
        } else if (markerCount > 1) {
            addV('MODEL_CARD_COUNT');
        } else {
            // Exactly 1 marker
            try {
                const card = JSON.parse(payloads[0]);
                if (typeof card !== 'object' || card === null || Array.isArray(card)) {
                    addV('INVALID_MODEL_CARD');
                } else {
                    // Match machine manifests and policy
                    const expectedFields = {
                        task: parsedManifest ? parsedManifest.task : undefined,
                        baseRevision: parsedManifest ? parsedManifest.base : undefined,
                        datasetDigest: parsedManifest ? parsedManifest.datasetDigest : undefined,
                        modelArtifactDigest: parsedManifest ? parsedManifest.modelArtifactDigest : undefined,
                        license: policy.license,
                        intendedUse: policy.intendedUse,
                        limitations: policy.limitations
                    };

                    for (const [key, expectedValue] of Object.entries(expectedFields)) {
                        if (expectedValue !== undefined && card[key] !== expectedValue) {
                            addV('MODEL_CARD_MISMATCH');
                            break;
                        }
                    }
                }
            } catch (e) {
                addV('INVALID_MODEL_CARD');
            }
        }
    }

    // 9. Format Response
    const violationsArray = Array.from(violations).sort();
    
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